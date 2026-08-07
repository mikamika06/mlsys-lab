def diagnose_plugin_issue(requested_plugin, registered_plugins):
    req_name = requested_plugin.get("name")
    req_ver = requested_plugin.get("version")
    req_ns = requested_plugin.get("namespace", "")
    req_fields = set(requested_plugin.get("fields", []))

    matching_names = [p for p in registered_plugins if p.get("name") == req_name]
    if not matching_names:
        return {
            "status": "MISSING_NAME",
            "reason": f"Plugin name '{req_name}' is not registered in the TRT Plugin Registry."
        }

    matching_ns = [p for p in matching_names if p.get("namespace", "") == req_ns]
    if not matching_ns:
        return {
            "status": "NAMESPACE_MISMATCH",
            "reason": f"Plugin '{req_name}' found, but namespace '{req_ns}' does not match."
        }

    matching_ver = [p for p in matching_ns if p.get("version") == req_ver]
    if not matching_ver:
        return {
            "status": "VERSION_MISMATCH",
            "reason": f"Plugin '{req_name}' in namespace '{req_ns}' found, but version '{req_ver}' is missing."
        }

    for p in matching_ver:
        reg_fields = set(p.get("fields", []))
        if req_fields.issubset(reg_fields):
            return {
                "status": "EXACT_MATCH",
                "reason": "Plugin found and registered successfully."
            }

    return {
        "status": "FIELD_MISMATCH",
        "reason": f"Plugin '{req_name}' found, but required fields are missing."
    }
