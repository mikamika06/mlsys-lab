def verify_index(index_data, available_files):
    weight_map = index_data.get("weight_map", {})
    referenced_files = set(weight_map.values())
    missing_files = referenced_files - set(available_files)
    if missing_files:
        return {"valid": False, "reason": f"missing files: {missing_files}"}

    assigned_weights = set(weight_map.keys())
    if not assigned_weights:
        return {"valid": False, "reason": "empty weight map"}

    return {"valid": True, "reason": "ok"}
