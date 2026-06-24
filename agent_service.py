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

SYSTEM_PROMPT = """You are the NautiCAI Data Review Copilot — an analyst assistant for side-scan sonar seabed survey triage.

You help desk reviewers understand detections, prioritize follow-up, and explain risk — grounded ONLY in tool results from the live session.

Rules:
- Always use tools to inspect session data before answering factual questions about contacts, counts, or risk.
- Never invent contact IDs, coordinates, confidence values, or clearance status.
- Do not issue clearance certificates or claim an area is safe for installation.
- Recommend triage actions for a human analyst to approve — you cannot change records.
- Use propose_lifecycle_transitions to suggest lifecycle steps (desk_review, rov_scheduled, rov_confirmed). Proposals require human approval; never claim you applied them.
- If the session has no frames or detections, say so and suggest loading frames in Data Review (/review).
- Be concise and operational. Cite contact IDs and frame numbers when discussing specific contacts.
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


@router.get("/health")
def agent_health():
    return {"ok": True, "model": DEFAULT_MODEL, "api_key_set": bool(os.environ.get("ANTHROPIC_API_KEY"))}


@router.post("/chat", response_model=ChatResponse)
def agent_chat(req: ChatRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set on the server.")
    try:
        import anthropic
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="pip install anthropic") from exc

    ctx = normalize_session(req.session, req.frames, req.survey_info, lifecycle=req.lifecycle)
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
            raise HTTPException(status_code=502, detail=f"Anthropic API error: {exc}") from exc

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
