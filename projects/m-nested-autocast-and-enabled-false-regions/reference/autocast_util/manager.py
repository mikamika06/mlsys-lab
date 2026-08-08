def get_effective_state(stack):
    if not stack:
        return {"enabled": False, "dtype": "fp32"}
    active_enabled = False
    active_dtype = "fp32"
    for item in stack:
        if item.get("explicit_disable"):
            active_enabled = False
            active_dtype = "fp32"
        else:
            if item.get("enabled") is not None:
                active_enabled = item["enabled"]
            if item.get("dtype") is not None:
                active_dtype = item["dtype"]
    return {"enabled": active_enabled, "dtype": active_dtype if active_enabled else "fp32"}
