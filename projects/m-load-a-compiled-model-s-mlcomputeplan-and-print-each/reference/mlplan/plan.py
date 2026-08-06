def parse_compute_plan(plan_dict):
    """Parse raw compute plan into structured list of op records."""
    ops = []
    raw_ops = plan_dict.get("operations", [])
    for op in raw_ops:
        dev = op.get("dispatched_device", "CPU")
        reason = op.get("ane_rejection_reason")
        if dev == "ANE":
            reason = None
        ops.append({
            "id": str(op.get("id", "")),
            "type": str(op.get("type", "")),
            "device": str(dev),
            "cost": float(op.get("estimated_cost", 0.0)),
            "ane_supported": bool(op.get("ane_supported", False)),
            "ane_rejection_reason": reason,
        })
    return ops
