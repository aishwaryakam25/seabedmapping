"""
ROV dive-planner — sequences rov_scheduled contacts into an ordered dive plan.

Verification-focused: per-dive briefs with suggested tooling. Never issues clearance.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/agent", tags=["agent"])

CLASS_TOOLING: dict[str, str] = {
    "wreck": "HD video + multibeam patch + scour profiler — assess extent and burial",
    "uxo": "Low-altitude HD survey + standoff per project EOD procedure",
    "mine": "Low-altitude HD survey + standoff per project EOD procedure",
    "pipeline": "Pipeline tracker + span/free-span check along contact axis",
    "cable": "Cable tracker + burial exposure check",
    "cable_crossing": "Crossing tracker + span and clearance measurement",
    "obstruction": "HD video + dimensioning — confirm identity and proud height",
    "boulder": "HD video + dimensioning — confirm size and burial",
    "rock": "HD video + dimensioning — confirm seabed contact",
    "debris": "HD video + grab sample if safe — classify material",
    "scour": "Scour profiler + multibeam patch — measure depth and extent",
    "anchor_drag": "HD video + seabed disturbance mapping",
}

DEFAULT_TOOLING = "HD video + dimensioning — confirm identity and extent"


def _norm_cls(cls: Any) -> str:
    return str(cls or "object").lower().replace(" ", "_")


def _tooling_for(cls: Any) -> str:
    return CLASS_TOOLING.get(_norm_cls(cls), DEFAULT_TOOLING)


def _location_label(contact: dict) -> str:
    parts = [f"Frame {contact.get('frame', '—')}"]
    if contact.get("range_m") is not None:
        parts.append(f"range ~{contact['range_m']} m")
    if contact.get("cxn") is not None and contact.get("cyn") is not None:
        parts.append(f"pos ({float(contact['cxn']):.2f}, {float(contact['cyn']):.2f})")
    if contact.get("east") is not None and contact.get("north") is not None:
        parts.append(f"E {contact['east']}, N {contact['north']}")
    return " · ".join(parts)


def _objective_for(contact: dict) -> str:
    cls = _norm_cls(contact.get("cls"))
    risk = contact.get("class_risk", "Medium")
    if cls in ("wreck", "uxo", "mine"):
        return f"Confirm identity and extent of {cls.replace('_', ' ')} contact — {risk} significance"
    if cls in ("pipeline", "cable", "cable_crossing"):
        return f"Verify {cls.replace('_', ' ')} condition, span, and burial along contact"
    if cls == "scour":
        return "Measure scour depth and extent relative to adjacent structure"
    return f"Confirm class, dimensions, and burial state of {cls.replace('_', ' ')} contact"


def _dive_notes(targets: list[dict]) -> list[str]:
    notes: list[str] = []
    if len(targets) > 1:
        notes.append(f"{len(targets)} co-located targets on this frame — single transit recommended")
    hi = [t for t in targets if t.get("class_risk") in ("Critical", "High")]
    if hi:
        notes.append(f"{len(hi)} high-significance target(s) — prioritise before lower-risk contacts")
    for t in targets:
        if float(t.get("conf") or 0) >= 0.85:
            notes.append(f"{t['id']}: high-confidence acoustic return ({int(float(t['conf']) * 100)}%)")
            break
    notes.append("Verification only — clearance remains with qualified surveyor")
    return notes


def _target_brief(contact: dict) -> dict:
    return {
        "contact_id": contact["id"],
        "cls": contact.get("cls"),
        "class_risk": contact.get("class_risk"),
        "conf": contact.get("conf"),
        "objective": _objective_for(contact),
        "tooling": _tooling_for(contact.get("cls")),
        "location": _location_label(contact),
    }


def build_dive_plan(contacts: list[dict[str, Any]]) -> dict[str, Any]:
    from agent_service import _risk_rank

    queue = [c for c in contacts if str(c.get("lifecycle") or "").lower() == "rov_scheduled"]
    if not queue:
        return {
            "verification_only": True,
            "clearance_not_issued": True,
            "empty": True,
            "summary": "No ROV-scheduled contacts — escalate targets in Triage first.",
            "dive_count": 0,
            "target_count": 0,
            "dives": [],
        }

    by_frame: dict[int, list[dict]] = {}
    for c in queue:
        fi = int(c.get("frame") or 1)
        by_frame.setdefault(fi, []).append(c)

    dives = []
    dive_num = 0
    for fi in sorted(by_frame.keys()):
        targets = sorted(
            by_frame[fi],
            key=lambda c: (-_risk_rank(c.get("class_risk", "Low")), -float(c.get("conf") or 0), c["id"]),
        )
        dive_num += 1
        briefs = [_target_brief(t) for t in targets]
        dives.append({
            "dive_number": dive_num,
            "frame": fi,
            "target_count": len(targets),
            "primary_risk": targets[0].get("class_risk"),
            "objective": (
                f"Frame {fi} verification pass — "
                + ", ".join(f"{t['id']} ({t.get('cls')})" for t in targets[:3])
                + ("…" if len(targets) > 3 else "")
            ),
            "location": _location_label(targets[0]),
            "tooling": "; ".join(dict.fromkeys(_tooling_for(t.get("cls")) for t in targets)),
            "targets": briefs,
            "notes": _dive_notes(targets),
        })

    n_targets = sum(d["target_count"] for d in dives)
    summary = (
        f"{len(dives)} dive{'s' if len(dives) != 1 else ''} planned across "
        f"{n_targets} ROV-scheduled contact{'s' if n_targets != 1 else ''} "
        f"(line order, highest priority first per frame)."
    )

    return {
        "verification_only": True,
        "clearance_not_issued": True,
        "empty": False,
        "summary": summary,
        "dive_count": len(dives),
        "target_count": n_targets,
        "dives": dives,
    }


def plan_to_text(plan: dict) -> str:
    if plan.get("empty"):
        return plan.get("summary", "No dives planned.")
    lines = ["NAUTICAI ROV DIVE PLAN", plan["summary"], ""]
    for d in plan.get("dives") or []:
        lines.append(f"DIVE {d['dive_number']} — Frame {d['frame']} ({d['target_count']} target(s))")
        lines.append(f"  Objective: {d['objective']}")
        lines.append(f"  Location: {d['location']}")
        lines.append(f"  Tooling: {d['tooling']}")
        for t in d.get("targets") or []:
            lines.append(
                f"    · {t['contact_id']} {t.get('cls')} [{t.get('class_risk')}] — {t.get('objective')}"
            )
        for note in d.get("notes") or []:
            lines.append(f"  Note: {note}")
        lines.append("")
    lines.append("Verification only — not a clearance certificate.")
    return "\n".join(lines).strip()


class DivePlanRequest(BaseModel):
    session: dict | None = None
    frames: list | None = None
    survey_info: dict | None = None
    lifecycle: dict | None = None


@router.post("/dive-plan")
def agent_dive_plan(req: DivePlanRequest):
    from agent_service import normalize_session

    ctx = normalize_session(
        req.session,
        req.frames,
        req.survey_info,
        lifecycle=req.lifecycle,
    )
    return build_dive_plan(ctx["contacts"])
