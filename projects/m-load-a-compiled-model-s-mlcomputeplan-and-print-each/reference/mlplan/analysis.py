def find_ane_rejections(parsed_ops):
    """Find ops rejected from ANE and return their details and cost information."""
    rejections = []
    for op in parsed_ops:
        if op.get("device") != "ANE" and op.get("ane_rejection_reason"):
            rejections.append({
                "id": op["id"],
                "type": op["type"],
                "device": op["device"],
                "cost": op["cost"],
                "reason": op["ane_rejection_reason"],
            })
    return rejections
