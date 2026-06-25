"""
Triage agent — bulk lifecycle proposal pass across the loaded survey.

Propose-only: groups contacts by recommended next action for human bulk approval.
Never proposes clearance.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from agent_actions import propose_lifecycle_transitions

router = APIRouter(prefix="/api/agent", tags=["agent"])

GROUP_ORDER = ("rov_scheduled", "desk_review", "rov_confirmed")
GROUP_LABELS: dict[str, str] = {
    "rov_scheduled": "Escalate to ROV",
    "desk_review": "Route to desk review",
    "rov_confirmed": "Mark ROV confirmed",
}


def build_triage_plan(
    contacts: list[dict[str, Any]],
    *,
    min_risk: str = "Low",
    limit: int = 500,
) -> dict[str, Any]:
    raw = propose_lifecycle_transitions(
        {"min_risk": min_risk, "limit": limit},
        contacts,
    )
    proposals = raw.get("proposals") or []

    buckets: dict[str, list[dict]] = {k: [] for k in GROUP_ORDER}
    risk_tally = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    for p in proposals:
        state = p.get("proposed_state")
        if state in buckets:
            buckets[state].append(p)
        risk = str(p.get("class_risk") or "Medium").title()
        if risk in risk_tally:
            risk_tally[risk] += 1

    groups = []
    parts = []
    for state in GROUP_ORDER:
        items = buckets[state]
        if not items:
            continue
        groups.append({
            "action": state,
            "label": GROUP_LABELS[state],
            "count": len(items),
            "proposals": items,
        })
        parts.append(f"{len(items)} {GROUP_LABELS[state].lower()}")

    total = len(proposals)
    if total == 0:
        summary = "No lifecycle transitions proposed — survey contacts are up to date or terminal."
    elif len(parts) == 1:
        summary = f"{total} contact{'s' if total != 1 else ''} proposed: {parts[0]}."
    else:
        summary = f"{total} contacts proposed: " + ", ".join(parts[:-1]) + f", and {parts[-1]}."

    return {
        "propose_only": True,
        "never_auto_applies": True,
        "clearance_not_proposed": True,
        "empty": total == 0,
        "summary": summary,
        "total": total,
        "skipped": raw.get("skipped", 0),
        "risk_tally": risk_tally,
        "groups": groups,
    }


class TriageRequest(BaseModel):
    session: dict | None = None
    frames: list | None = None
    survey_info: dict | None = None
    lifecycle: dict | None = None
    min_risk: str = Field(default="Low")
    limit: int = Field(default=500, ge=1, le=2000)


@router.post("/triage")
def agent_triage(req: TriageRequest):
    from agent_service import normalize_session

    ctx = normalize_session(
        req.session,
        req.frames,
        req.survey_info,
        lifecycle=req.lifecycle,
    )
    return build_triage_plan(ctx["contacts"], min_risk=req.min_risk, limit=req.limit)
