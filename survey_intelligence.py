"""Backend helpers for survey intelligence features (clearance PDF, exports)."""


def clearance_contacts_from_meta(meta):
    """Extract contacts eligible for clearance from PDF meta payload."""
    meta = meta or {}
    lifecycle = meta.get("lifecycle") or {}
    items = lifecycle.get("items") if isinstance(lifecycle, dict) else lifecycle
    if not items:
        items = {}
    if isinstance(items, list):
        mapped = {}
        for it in items:
            if isinstance(it, dict) and it.get("id"):
                mapped[it["id"]] = it
        items = mapped
    cleared = []
    for cid, rec in items.items():
        state = (rec or {}).get("state") or ""
        if state in ("rov_confirmed", "clearance_issued"):
            cleared.append({"contact_id": cid, **rec})
    return cleared


def intelligence_summary(meta):
    """Build a short summary dict for PDF / API responses."""
    meta = meta or {}
    roi = meta.get("intelligence_roi") or {}
    fused = meta.get("fused_contacts") or []
    temporal = meta.get("temporal_diff") or {}
    return {
        "clearance_count": len(clearance_contacts_from_meta(meta)),
        "dives_avoided": roi.get("dives_avoided", 0),
        "cost_saved_usd": roi.get("cost_saved_usd", 0),
        "co2_saved_kg": roi.get("co2_saved_kg", 0),
        "fused_contacts": len(fused),
        "multi_line_confirmed": sum(1 for f in fused if (f.get("corroboration") or 0) > 1),
        "new_contacts_4d": temporal.get("new_contacts", 0),
        "removed_contacts_4d": temporal.get("removed_contacts", 0),
        "install_date": meta.get("install_date") or "",
    }
