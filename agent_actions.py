"""
Agent action tools and briefing narrative — extends the Data Review Copilot loop.

Safety: propose-only lifecycle transitions; briefing never issues GO/NO-GO or clearance.
"""
from __future__ import annotations

import json
import os
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/agent", tags=["agent"])

DEFAULT_MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-4-6")

TERMINAL_LIFECYCLE = frozenset({"rov_confirmed", "clearance_issued", "cleared"})
NEVER_PROPOSE = frozenset({"clearance_issued", "cleared"})

PROPOSE_TRANSITIONS_TOOL = {
    "name": "propose_lifecycle_transitions",
    "description": (
        "Propose lifecycle transitions (propose-only — analyst must approve). "
        "High-significance ai_detected → rov_scheduled; lower → desk_review; "
        "rov_scheduled → rov_confirmed. Never proposes clearance."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "min_risk": {"type": "string", "description": "Critical, High, Medium, or Low (default Low)."},
            "limit": {"type": "integer"},
            "contact_ids": {"type": "array", "items": {"type": "string"}},
        },
    },
}

BRIEFING_SYSTEM = """You write client-ready seabed survey briefing prose for offshore projects.

Rules:
- Describe survey scope, key findings, and contact classes — grounded only in provided data.
- Recommend ROV verification or desk review for high-significance open contacts.
- Do NOT issue GO, NO-GO, or CONDITIONAL safety decisions.
- Do NOT claim the site is cleared or safe for installation.
- Label ROI/remediation figures as indicative estimates.
- Write 2–4 short paragraphs. End reminding final clearance rests with a qualified surveyor.
"""


def _lifecycle_state(contact: dict) -> str:
    return str(contact.get("lifecycle") or "ai_detected").strip().lower()


def _propose_next(contact: dict) -> tuple[str, str] | None:
    lc = _lifecycle_state(contact)
    status = contact.get("status") or "Review"
    risk = contact.get("class_risk") or "Medium"
    if lc in TERMINAL_LIFECYCLE or status == "Rejected" or lc in NEVER_PROPOSE:
        return None
    if lc == "ai_detected":
        if risk in ("Critical", "High"):
            return "rov_scheduled", f"{risk} significance ({contact.get('cls')}) — escalate to ROV scheduling"
        return "desk_review", f"{risk} significance — route to desk review"
    if lc == "desk_review" and risk in ("Critical", "High"):
        return "rov_scheduled", f"Open {risk} contact — recommend ROV scheduling"
    if lc == "rov_scheduled":
        return "rov_confirmed", "ROV scheduled — propose confirmation after field verification"
    return None


def propose_lifecycle_transitions(args: dict, contacts: list[dict]) -> dict:
    from agent_service import _risk_rank

    min_risk = str(args.get("min_risk") or "Low").strip().title()
    min_rank = _risk_rank(min_risk)
    limit = int(args.get("limit") or 15)
    id_filter = args.get("contact_ids")
    pool = contacts
    if id_filter:
        needles = {str(x).strip().lower() for x in id_filter}
        pool = [c for c in contacts if str(c.get("id", "")).lower() in needles]

    proposals, skipped = [], 0
    for contact in pool:
        if _risk_rank(contact.get("class_risk", "Medium")) < min_rank:
            continue
        nxt = _propose_next(contact)
        if not nxt:
            skipped += 1
            continue
        proposed_state, reason = nxt
        if proposed_state in NEVER_PROPOSE:
            skipped += 1
            continue
        proposals.append({
            "contact_id": contact["id"], "cls": contact.get("cls"), "frame": contact.get("frame"),
            "class_risk": contact.get("class_risk"), "current_lifecycle": _lifecycle_state(contact),
            "proposed_state": proposed_state, "reason": reason, "requires_human_approval": True,
        })

    proposals.sort(key=lambda p: (-_risk_rank(p.get("class_risk", "Low")), p["contact_id"]))
    return {
        "propose_only": True, "never_auto_applies": True, "clearance_not_proposed": True,
        "count": len(proposals[:limit]), "skipped": skipped, "proposals": proposals[:limit],
    }


def _anthropic_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set on the server.")
    try:
        import anthropic
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="pip install anthropic") from exc
    return anthropic.Anthropic(api_key=api_key)


def generate_briefing_narrative(
    *, session=None, frames=None, survey_info=None, lifecycle=None, roi=None, remediation=None,
) -> str:
    from agent_service import normalize_session, tool_survey_summary

    ctx = normalize_session(session, frames, survey_info, lifecycle=lifecycle)
    summary = tool_survey_summary(ctx)
    payload = {
        "survey": summary["survey"], "totals": summary["totals"],
        "by_class": summary["by_class"], "by_class_risk": summary["by_class_risk"],
        "by_status": summary["by_status"],
        "roi_indicative": roi or {}, "remediation_indicative": remediation or {},
    }
    client = _anthropic_client()
    try:
        response = client.messages.create(
            model=DEFAULT_MODEL, max_tokens=2048, system=BRIEFING_SYSTEM,
            messages=[{"role": "user", "content": "Write briefing prose from this data (no GO/NO-GO):\n\n" + json.dumps(payload, indent=2)}],
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Anthropic API error: {exc}") from exc
    parts = [b.text for b in response.content if hasattr(b, "text") and b.text]
    return "\n\n".join(parts).strip() or "No narrative generated."


def generate_survey_executive_narrative(meta, *, frame_count, detection_count, high_risk_count, class_counts, avg_confidence):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        session = {
            "frames": frame_count, "hasData": detection_count > 0,
            "project": meta.get("project"), "location": meta.get("location"),
            "vessel": meta.get("vessel"), "sensor": meta.get("sensor"),
            "detections": [{"cls": cls, "conf": avg_confidence, "frame": 1, "status": "Review"}
                           for cls, count in class_counts.items() for _ in range(count)],
        }
        return generate_briefing_narrative(session=session, lifecycle=meta.get("lifecycle"),
                                           roi=meta.get("intelligence_roi"), remediation=meta.get("remediation"))
    except Exception:
        return None


class BriefingRequest(BaseModel):
    session: dict | None = None
    frames: list | None = None
    survey_info: dict | None = None
    lifecycle: dict | None = None
    roi: dict | None = None
    remediation: dict | None = None


class BriefingResponse(BaseModel):
    narrative: str


@router.post("/briefing", response_model=BriefingResponse)
def agent_briefing(req: BriefingRequest):
    return BriefingResponse(narrative=generate_briefing_narrative(
        session=req.session, frames=req.frames, survey_info=req.survey_info,
        lifecycle=req.lifecycle, roi=req.roi, remediation=req.remediation,
    ))
