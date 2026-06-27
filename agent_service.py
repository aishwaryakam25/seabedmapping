"""
Data Review Copilot — agent service with tool use over live survey sessions.

    set ANTHROPIC_API_KEY=sk-ant-...
    set AGENT_MODEL=claude-sonnet-4-6   # optional
"""
from __future__ import annotations

import json
import os
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agent_actions import PROPOSE_TRANSITIONS_TOOL, propose_lifecycle_transitions

router = APIRouter(prefix="/api/agent", tags=["agent"])

DEFAULT_MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-4-6")
MAX_ITERATIONS = 8

# Free local fallback (Ollama). Used automatically when no cloud backend is available.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("AGENT_OLLAMA_MODEL", "llama3.2:1b")

# Free cloud backend (Groq, OpenAI-compatible). Fast; free API key, no card required.
GROQ_URL = os.environ.get("GROQ_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# Backend selection: "auto" (claude → groq → local), or force one of
# "claude" / "groq" / "local". Each cloud backend falls back to local on failure.
AGENT_BACKEND = os.environ.get("AGENT_BACKEND", "auto").lower()

CLASS_RISK_LABEL: dict[str, str] = {
    "wreck": "Critical", "uxo": "Critical", "mine": "Critical",
    "pipeline": "High", "cable": "High", "cable_crossing": "High", "obstruction": "High",
    "boulder": "Medium", "anchor": "Medium", "debris": "Medium", "scour": "Medium", "rock": "Low",
}

CLASS_RISK_WEIGHT: dict[str, float] = {
    "wreck": 1.0, "uxo": 1.0, "mine": 1.0,
    "pipeline": 0.88, "cable": 0.85, "cable_crossing": 0.85, "obstruction": 0.82,
    "boulder": 0.62, "anchor": 0.58, "debris": 0.55, "scour": 0.48, "rock": 0.35,
}

HIGH_T, MED_T = 0.65, 0.35
FIELD_CLS = ("cls", "class", "label")
FIELD_CONF = ("conf", "confidence", "score")
FIELD_ID = ("contact_id", "contactId", "id")
FIELD_FRAME = ("frame", "frameIndex", "fi")

SYSTEM_PROMPT = """You are NautiCAI Chatbot — a friendly assistant that helps analysts during side-scan sonar seabed-mapping sessions, answering their questions and clearing their doubts.

You handle two kinds of questions:
1. Concept / how-to questions — explain from your own knowledge: how side-scan sonar works, how to read an acoustic shadow, the difference between a wreck, UXO, debris or a boulder, what a risk tier or lifecycle state means, what "burial" or "proud on seabed" implies, how to use the Data Review / Triage / Dive Plan screens. Answer these directly, in plain language, and end with a practical takeaway.
2. Questions about the survey loaded right now — answer from the survey information available to you (counts, highest-risk contacts, a specific contact, follow-up suggestions). Never invent contact IDs, counts, coordinates, confidence values, or status. If something isn't in the data, say it isn't available and suggest loading frames in Data Review.

Voice — follow strictly:
- Refer to yourself only as "NautiCAI Chatbot", or just "I". NEVER call yourself a "copilot" or an "AI model".
- Speak in plain, operational language for a survey analyst. NEVER mention internal or technical implementation details: no function or tool names, no model names or versions (in particular never say "YOLO"), no file names, no code, no software jargon about how you work.
- Be concise and friendly. When discussing specific contacts, cite their IDs and frame numbers.

Safety:
- You can suggest a next step for a contact — route it to desk review, escalate it to ROV, or mark it ROV-confirmed — for a person to approve. You cannot change records yourself, and must never claim you applied a change.
- Never issue a clearance or state that an area is safe for installation; that decision belongs to a qualified surveyor. Mention this only briefly when relevant — do not preface every answer with a disclaimer.
"""

TOOL_DEFINITIONS = [
    {
        "name": "survey_summary",
        "description": "High-level overview: frame count, detections, class/risk/status breakdown, survey metadata.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "query_contacts",
        "description": "Query contacts with optional filters (risk, class, frame, min_conf, status).",
        "input_schema": {
            "type": "object",
            "properties": {
                "risk": {"type": "string"},
                "cls": {"type": "string"},
                "frame": {"type": "integer"},
                "min_conf": {"type": "number"},
                "status": {"type": "string"},
                "limit": {"type": "integer"},
            },
        },
    },
    {
        "name": "explain_contact",
        "description": "Explain why a contact received its risk tier (acoustic cues, burial estimate).",
        "input_schema": {
            "type": "object",
            "properties": {"contact_id": {"type": "string"}},
            "required": ["contact_id"],
        },
    },
    {
        "name": "recommend_followup",
        "description": "High-significance contacts for ROV/desk follow-up, ranked by risk score.",
        "input_schema": {
            "type": "object",
            "properties": {
                "min_risk": {"type": "string"},
                "limit": {"type": "integer"},
            },
        },
    },
    PROPOSE_TRANSITIONS_TOOL,
]


def _norm_key(value: Any) -> str:
    return str(value or "").lower().replace(" ", "_")


def _pick(obj: dict | None, keys: tuple[str, ...], fallback: Any = None) -> Any:
    if not obj:
        return fallback
    for key in keys:
        val = obj.get(key)
        if val is not None and val != "":
            return val
    return fallback


def _norm_conf(value: Any) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return 0.0
    return n / 100.0 if n > 1 else n


def class_risk_label(cls: Any) -> str:
    return CLASS_RISK_LABEL.get(_norm_key(cls), "Medium")


def risk_score(cls: Any, conf: Any) -> float:
    return round(CLASS_RISK_WEIGHT.get(_norm_key(cls), 0.5) * _norm_conf(conf), 4)


def score_tier(cls: Any, conf: Any) -> str:
    s = risk_score(cls, conf)
    if s >= HIGH_T:
        return "High"
    if s >= MED_T:
        return "Medium"
    return "Low"


def _risk_rank(label: str) -> int:
    return {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}.get(label, 0)


def shadow_metrics(det: dict, nat_w: float = 1024, nat_h: float = 1257, range_scale: float = 150) -> dict:
    box = det.get("box") or [0, 0, 0, 0]
    w = max(1.0, float(box[2]) - float(box[0]))
    h = max(1.0, float(box[3]) - float(box[1]))
    wnorm = det.get("wnorm") if det.get("wnorm") is not None else w / max(1.0, nat_w)
    aspect = h / w
    shadow_len_norm = min(0.45, aspect * 0.35 + float(wnorm) * 0.25)
    height_m = shadow_len_norm * range_scale * 0.55
    if aspect < 0.55 and height_m < 0.35:
        burial, burial_label = "buried", "Likely buried"
    elif height_m < 0.8:
        burial, burial_label = "partially_buried", "Partially buried"
    else:
        burial, burial_label = "proud", "Proud on seabed"
    return {
        "height_m": round(height_m, 2),
        "shadow_len_m": round(shadow_len_norm * range_scale, 2),
        "burial": burial,
        "burial_label": burial_label,
    }


def explain_contact_logic(det: dict, nat_w: float = 1024, nat_h: float = 1257, range_scale: float = 150) -> dict:
    box = det.get("box") or [0, 0, 0, 0]
    w = float(box[2]) - float(box[0])
    h = float(box[3]) - float(box[1])
    aspect = h / max(1.0, w)
    cues: list[str] = []
    if aspect > 1.2:
        cues.append("elongated acoustic shadow")
    if aspect < 0.7:
        cues.append("compact bright return")
    if _norm_conf(det.get("conf")) > 0.85:
        cues.append("strong hyperbolic apex")
    cues.append("broad seabed contact" if w / max(1.0, nat_w) > 0.08 else "isolated point target")
    physics = shadow_metrics(det, nat_w, nat_h, range_scale)
    cues.append(physics["burial_label"].lower())
    cls = _pick(det, FIELD_CLS, "unknown")
    return {
        "contact_id": det.get("id"),
        "cls": cls,
        "conf": _norm_conf(det.get("conf")),
        "class_risk": class_risk_label(cls),
        "score_tier": score_tier(cls, det.get("conf")),
        "risk_score": risk_score(cls, det.get("conf")),
        "summary": " · ".join(cues[:3]),
        "cues": cues,
        "physics": physics,
        "status": det.get("status") or "Review",
        "frame": det.get("frame"),
        "lifecycle": det.get("lifecycle"),
    }


def _contact_from_det(
    det: dict, frame_idx: int, det_idx: int,
    nat_w: float | None = None, nat_h: float | None = None,
    range_scale: float = 150, lifecycle_items: dict | None = None,
) -> dict:
    cls = _pick(det, FIELD_CLS, "unknown")
    conf = _norm_conf(_pick(det, FIELD_CONF, 0))
    cid = _pick(det, FIELD_ID, None) or f"F{frame_idx}-D{det_idx + 1}"
    range_m = det.get("range")
    if range_m is None and det.get("cxn") is not None:
        range_m = abs(float(det["cxn"]) - 0.5) * 2 * range_scale
    lc = det.get("lifecycle")
    if not lc and lifecycle_items:
        rec = lifecycle_items.get(cid) or lifecycle_items.get(str(cid))
        if isinstance(rec, dict):
            lc = rec.get("state")
    return {
        "id": cid, "cls": cls, "conf": conf, "frame": frame_idx,
        "status": det.get("status") or "Review",
        "lifecycle": lc or "ai_detected",
        "box": det.get("box"), "cxn": det.get("cxn"), "cyn": det.get("cyn"),
        "wnorm": det.get("wnorm"), "east": det.get("east"), "north": det.get("north"),
        "range_m": round(float(range_m), 1) if range_m is not None else None,
        "note": det.get("note") or "", "burial": det.get("burial"),
        "height_m": det.get("height_m"), "urgency": det.get("urgency"),
        "class_risk": class_risk_label(cls), "score_tier": score_tier(cls, conf),
        "risk_score": risk_score(cls, conf), "nat_w": nat_w, "nat_h": nat_h,
    }


def normalize_session(
    session: dict | None = None, frames: list | None = None,
    survey_info: dict | None = None, lifecycle: dict | None = None,
) -> dict[str, Any]:
    session, frames, survey_info = session or {}, frames or [], survey_info or {}
    lifecycle_items: dict = {}
    if isinstance(lifecycle, dict) and isinstance(lifecycle.get("items"), dict):
        lifecycle_items = lifecycle["items"]

    range_scale = 150
    for src in (survey_info, session):
        raw = src.get("range")
        if raw is not None:
            try:
                range_scale = int(str(raw).replace("m", "").strip())
            except ValueError:
                pass

    contacts, frame_meta = [], []
    if frames:
        for fi, fr in enumerate(frames):
            nat_w, nat_h = fr.get("natW") or fr.get("nat_w"), fr.get("natH") or fr.get("nat_h")
            dets = fr.get("dets") or fr.get("detections") or []
            frame_meta.append({"frame": fi + 1, "name": fr.get("name") or f"Frame {fi + 1}",
                                 "detection_count": len(dets), "nat_w": nat_w, "nat_h": nat_h})
            for di, det in enumerate(dets):
                contacts.append(_contact_from_det(det, fi + 1, di, nat_w, nat_h, range_scale, lifecycle_items))
    elif session.get("detections"):
        n_frames = int(session.get("frames") or 1)
        frame_meta = [{"frame": i + 1, "name": f"Frame {i + 1}", "detection_count": 0} for i in range(n_frames)]
        for di, det in enumerate(session["detections"]):
            contacts.append(_contact_from_det(
                det, int(_pick(det, FIELD_FRAME, 1) or 1), di,
                range_scale=range_scale, lifecycle_items=lifecycle_items,
            ))
        for fr in frame_meta:
            fr["detection_count"] = sum(1 for c in contacts if c["frame"] == fr["frame"])

    meta = {
        "project": survey_info.get("project") or session.get("project") or "",
        "survey_line": survey_info.get("survey_line") or session.get("survey_line") or "",
        "date": survey_info.get("date") or session.get("date") or "",
        "location": survey_info.get("location") or session.get("location") or "",
        "vessel": survey_info.get("vessel") or session.get("vessel") or "",
        "sensor": survey_info.get("sensor") or session.get("sensor") or "",
        "range_m": range_scale,
        "frames": len(frame_meta) or int(session.get("frames") or 0),
        "has_data": bool(contacts) or bool(frame_meta),
    }
    return {"meta": meta, "frames": frame_meta, "contacts": contacts}


def tool_survey_summary(ctx: dict) -> dict:
    contacts, meta = ctx["contacts"], ctx["meta"]
    by_class, by_risk, by_status = {}, {}, {}
    for c in contacts:
        by_class[c["cls"]] = by_class.get(c["cls"], 0) + 1
        by_risk[c["class_risk"]] = by_risk.get(c["class_risk"], 0) + 1
        st = c.get("status") or "Review"
        by_status[st] = by_status.get(st, 0) + 1
    high_open = sum(1 for c in contacts if c["class_risk"] in ("Critical", "High")
                    and c.get("status") not in ("Verified", "Rejected"))
    return {
        "survey": meta,
        "totals": {"frames": meta["frames"], "detections": len(contacts), "high_risk_open": high_open},
        "by_class": by_class, "by_class_risk": by_risk, "by_status": by_status,
        "frames": ctx["frames"],
    }


def tool_query_contacts(ctx: dict, args: dict) -> dict:
    contacts = list(ctx["contacts"])
    if args.get("risk"):
        rf = str(args["risk"]).strip().title()
        contacts = [c for c in contacts if c["class_risk"] == rf]
    if args.get("cls"):
        ck = _norm_key(args["cls"])
        contacts = [c for c in contacts if _norm_key(c["cls"]) == ck]
    if args.get("frame") is not None:
        contacts = [c for c in contacts if c["frame"] == int(args["frame"])]
    if args.get("min_conf") is not None:
        contacts = [c for c in contacts if c["conf"] >= float(args["min_conf"])]
    if args.get("status"):
        contacts = [c for c in contacts if (c.get("status") or "Review") == args["status"]]
    contacts.sort(key=lambda c: (-c["risk_score"], c["frame"], c["id"]))
    limit = int(args.get("limit") or 20)
    trimmed = contacts[:limit]
    return {
        "matched": len(contacts), "returned": len(trimmed),
        "contacts": [{k: c[k] for k in ("id", "cls", "conf", "frame", "status", "class_risk",
                                          "score_tier", "risk_score", "range_m", "lifecycle")} for c in trimmed],
    }


def _find_contact(ctx: dict, contact_id: str) -> dict | None:
    needle = str(contact_id).strip().lower()
    for c in ctx["contacts"]:
        if str(c["id"]).lower() == needle:
            return c
    return None


def tool_explain_contact(ctx: dict, args: dict) -> dict:
    cid = args.get("contact_id")
    if not cid:
        return {"error": "contact_id is required"}
    det = _find_contact(ctx, cid)
    if not det:
        return {"error": f"Contact {cid!r} not found"}
    return explain_contact_logic(det, float(det.get("nat_w") or 1024),
                                 float(det.get("nat_h") or 1257), float(ctx["meta"].get("range_m") or 150))


def tool_recommend_followup(ctx: dict, args: dict) -> dict:
    min_rank = _risk_rank(str(args.get("min_risk") or "High").strip().title())
    limit = int(args.get("limit") or 10)
    candidates = [c for c in ctx["contacts"] if _risk_rank(c["class_risk"]) >= min_rank
                  and c.get("status") not in ("Verified", "Rejected")]
    candidates.sort(key=lambda c: (-c["risk_score"], -_risk_rank(c["class_risk"]), c["frame"]))
    recs = []
    for c in candidates[:limit]:
        action = "Schedule ROV inspection" if c["class_risk"] in ("Critical", "High") else "Desk review / verify"
        recs.append({
            "contact_id": c["id"], "cls": c["cls"], "conf": c["conf"], "frame": c["frame"],
            "class_risk": c["class_risk"], "risk_score": c["risk_score"], "status": c.get("status"),
            "lifecycle": c.get("lifecycle"), "recommended_action": action,
            "rationale": f"{c['class_risk']} significance class with score tier {c['score_tier']}",
        })
    return {"count": len(recs), "recommendations": recs}


def execute_tool(name: str, args: dict, ctx: dict) -> dict:
    if name == "survey_summary":
        return tool_survey_summary(ctx)
    if name == "query_contacts":
        return tool_query_contacts(ctx, args or {})
    if name == "explain_contact":
        return tool_explain_contact(ctx, args or {})
    if name == "recommend_followup":
        return tool_recommend_followup(ctx, args or {})
    if name == "propose_lifecycle_transitions":
        return propose_lifecycle_transitions(args or {}, ctx["contacts"])
    return {"error": f"Unknown tool: {name}"}


def _extract_text(response: Any) -> str:
    return "\n".join(b.text for b in response.content if hasattr(b, "text") and b.text).strip() or "No response."


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    history: list[ChatMessage] = Field(default_factory=list)
    session: dict | None = None
    frames: list | None = None
    survey_info: dict | None = None
    lifecycle: dict | None = None


class ChatResponse(BaseModel):
    reply: str
    tools_used: list[str] = Field(default_factory=list)
    proposals: list[dict[str, Any]] = Field(default_factory=list)


class _BackendUnavailable(Exception):
    """Raised when a chat backend can't serve the request, so we can fall back."""


def _has_real_key() -> bool:
    key = os.environ.get("ANTHROPIC_API_KEY") or ""
    return bool(key) and not key.rstrip().endswith("...")


def _ollama_available() -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(OLLAMA_URL.rstrip("/") + "/api/tags", timeout=2):
            return True
    except Exception:
        return False


@router.get("/health")
def agent_health():
    return {
        "ok": True,
        "backend": AGENT_BACKEND,
        "claude_model": DEFAULT_MODEL,
        "claude_key_set": _has_real_key(),
        "groq_model": GROQ_MODEL,
        "groq_key_set": bool(os.environ.get("GROQ_API_KEY")),
        "local_model": OLLAMA_MODEL,
        "local_available": _ollama_available(),
    }


def _claude_chat(req: ChatRequest, ctx: dict) -> ChatResponse:
    """Full Claude tool-use loop. Raises _BackendUnavailable so the caller can fall back."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not _has_real_key():
        raise _BackendUnavailable("no Anthropic API key")
    try:
        import anthropic
    except ImportError as exc:
        raise _BackendUnavailable("anthropic package not installed") from exc

    client = anthropic.Anthropic(api_key=api_key)
    messages: list[dict] = []
    proposals: list[dict] = []
    tools_used: list[str] = []
    for item in req.history[-12:]:
        if item.role in ("user", "assistant"):
            messages.append({"role": item.role, "content": item.content})
    messages.append({"role": "user", "content": req.message})

    for _ in range(MAX_ITERATIONS):
        try:
            response = client.messages.create(
                model=DEFAULT_MODEL, max_tokens=4096, system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS, messages=messages,
            )
        except anthropic.APIError as exc:
            # auth, credit, rate-limit, etc. — let the caller fall back to local
            raise _BackendUnavailable(f"Anthropic API error: {exc}") from exc

        if response.stop_reason == "end_turn":
            return ChatResponse(reply=_extract_text(response), tools_used=tools_used, proposals=proposals)

        tool_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_blocks:
            return ChatResponse(reply=_extract_text(response), tools_used=tools_used, proposals=proposals)

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in tool_blocks:
            tools_used.append(block.name)
            result = execute_tool(block.name, block.input or {}, ctx)
            if block.name == "propose_lifecycle_transitions":
                proposals.extend(result.get("proposals") or [])
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)})
        messages.append({"role": "user", "content": tool_results})

    return ChatResponse(
        reply="I reached the maximum reasoning steps. Please try a more specific question.",
        tools_used=tools_used, proposals=proposals,
    )


def _survey_digest(ctx: dict) -> str:
    """Compact, grounded text summary of the live survey for the local model."""
    summ = tool_survey_summary(ctx)
    meta = summ["survey"]
    n = summ["totals"]["detections"]
    lines = [
        f"Project: {meta.get('project') or 'n/a'} | Line: {meta.get('survey_line') or 'n/a'} | "
        f"Frames: {meta.get('frames')} | Detections: {n} | Range scale: {meta.get('range_m')} m"
    ]
    if n == 0:
        lines.append("No detections are loaded in the current session. "
                     "Suggest the user load frames in Data Review (/review).")
        return "\n".join(lines)
    lines.append("Counts by class: " + ", ".join(f"{k}×{v}" for k, v in summ["by_class"].items()))
    lines.append("Counts by risk tier: " + ", ".join(f"{k}×{v}" for k, v in summ["by_class_risk"].items()))
    recs = tool_recommend_followup(ctx, {"min_risk": "Medium", "limit": 10})
    if recs["recommendations"]:
        lines.append("Highest-priority contacts (for follow-up):")
        for r in recs["recommendations"]:
            lines.append(
                f"  - {r['contact_id']}: {r['cls']} [{r['class_risk']}], frame {r['frame']}, "
                f"conf {int(float(r['conf']) * 100)}% — {r['recommended_action']}"
            )
    return "\n".join(lines)


def _grounded_messages(req: ChatRequest, ctx: dict) -> list[dict]:
    """Build chat messages for a no-tool backend: persona + survey digest + history + message."""
    system = (
        SYSTEM_PROMPT
        + "\n\nUse ONLY the survey information below to answer questions about the loaded survey; "
        "if something isn't in it, say it isn't available. Do not invent contact IDs, counts, or "
        "coordinates. Keep answers SHORT and direct — at most 4 sentences or 5 short bullet points. "
        "Lead with the answer, no preamble. Remember the Voice rules: plain language only, never "
        "mention tools, model names, or how you work.\n\n"
        "--- Survey information ---\n" + _survey_digest(ctx)
    )
    messages = [{"role": "system", "content": system}]
    for item in req.history[-6:]:
        if item.role in ("user", "assistant"):
            messages.append({"role": item.role, "content": item.content})
    messages.append({"role": "user", "content": req.message})
    return messages


def _groq_chat(req: ChatRequest, ctx: dict) -> ChatResponse:
    """Free, fast cloud backend via Groq (OpenAI-compatible). Falls back if no key/quota."""
    import urllib.request

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise _BackendUnavailable("no Groq API key")
    payload = json.dumps({
        "model": GROQ_MODEL,
        "messages": _grounded_messages(req, ctx),
        "max_tokens": 400,
        "temperature": 0.4,
    }).encode()
    request = urllib.request.Request(
        GROQ_URL, data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as resp:
            out = json.loads(resp.read())
    except Exception as exc:
        # bad key, rate limit, network — fall back to local
        raise _BackendUnavailable(f"Groq error: {exc}") from exc
    reply = ((out.get("choices") or [{}])[0].get("message") or {}).get("content", "").strip()
    if not reply:
        raise _BackendUnavailable("Groq returned no content")
    return ChatResponse(reply=reply, tools_used=["cloud:groq"], proposals=[])


def _ollama_chat(req: ChatRequest, ctx: dict) -> ChatResponse:
    """Free local fallback via Ollama. Grounds answers with a survey digest (no tool-calling)."""
    import urllib.request

    messages = _grounded_messages(req, ctx)
    payload = json.dumps({
        "model": OLLAMA_MODEL, "messages": messages, "stream": False,
        "keep_alive": "30m",       # keep the model resident so replies after the first are fast
        "options": {"num_predict": 200, "num_ctx": 4096},  # cap output length → faster replies
    }).encode()
    url = OLLAMA_URL.rstrip("/") + "/api/chat"
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=180) as resp:
            out = json.loads(resp.read())
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(f"Local model (Ollama) unavailable: {exc}. Make sure the Ollama app is running "
                    f"and the model '{OLLAMA_MODEL}' is installed."),
        ) from exc
    reply = ((out.get("message") or {}).get("content") or "").strip() or "No response."
    return ChatResponse(reply=reply, tools_used=[f"local:{OLLAMA_MODEL}"], proposals=[])


@router.post("/chat", response_model=ChatResponse)
def agent_chat(req: ChatRequest):
    ctx = normalize_session(req.session, req.frames, req.survey_info, lifecycle=req.lifecycle)

    if AGENT_BACKEND == "local":
        return _ollama_chat(req, ctx)
    if AGENT_BACKEND == "claude":
        try:
            return _claude_chat(req, ctx)
        except _BackendUnavailable as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    if AGENT_BACKEND == "groq":
        try:
            return _groq_chat(req, ctx)
        except _BackendUnavailable:
            return _ollama_chat(req, ctx)  # fall back to local

    # auto: Claude (if key+credit) → Groq (if key) → local. First one that works wins.
    for backend in (_claude_chat, _groq_chat):
        try:
            return backend(req, ctx)
        except _BackendUnavailable:
            continue
    return _ollama_chat(req, ctx)


# ---------------------------------------------------------------------------
# Spoken survey briefing — a grounded, deterministic narration the UI reads aloud.
# Numbers come straight from the survey tools, so it never hallucinates on a demo.
# ---------------------------------------------------------------------------
def build_briefing(ctx: dict) -> dict[str, Any]:
    from agent_triage import build_triage_plan

    summ = tool_survey_summary(ctx)
    meta = summ["survey"]
    total = summ["totals"]["detections"]
    if total == 0:
        return {
            "empty": True,
            "briefing": "No survey is loaded yet. Load side-scan frames in Data Review, "
                        "and I'll brief you on the contacts.",
            "highlights": [], "risk_tally": {},
        }

    risk = summ["by_class_risk"]
    order = ["Critical", "High", "Medium", "Low"]
    recs = tool_recommend_followup(ctx, {"min_risk": "Medium", "limit": 5})["recommendations"]
    height_by_id = {c["id"]: c.get("height_m") for c in ctx["contacts"]}

    frames = meta.get("frames") or 1
    project = meta.get("project") or "this survey line"
    parts = [f"Survey briefing for {project}."]
    parts.append(
        f"{total} contact{'s' if total != 1 else ''} detected across "
        f"{frames} frame{'s' if frames != 1 else ''}."
    )
    tally = [f"{risk[k]} {k.lower()}" for k in order if risk.get(k)]
    if tally:
        parts.append("Risk breakdown: " + ", ".join(tally) + ".")

    if recs:
        top = recs[0]
        h = height_by_id.get(top["contact_id"])
        ht = f", standing about {h} metres proud of the seabed" if h else ""
        parts.append(
            f"Top priority is {top['contact_id']}, a {top['cls']} on frame {top['frame']}, "
            f"{str(top['class_risk']).lower()} significance{ht}."
        )
        if len(recs) > 1:
            others = ", ".join(r["contact_id"] for r in recs[1:4])
            parts.append(f"Also flagged for follow-up: {others}.")

    triage = build_triage_plan(ctx["contacts"])
    if not triage.get("empty"):
        parts.append(triage["summary"])
    parts.append(
        "No area is cleared for installation; that decision stays with a qualified surveyor."
    )

    return {
        "empty": False,
        "briefing": " ".join(parts),
        "highlights": [r["contact_id"] for r in recs],
        "risk_tally": {k: risk.get(k, 0) for k in order},
    }


@router.post("/briefing")
def agent_briefing(req: ChatRequest):
    ctx = normalize_session(req.session, req.frames, req.survey_info, lifecycle=req.lifecycle)
    return build_briefing(ctx)
