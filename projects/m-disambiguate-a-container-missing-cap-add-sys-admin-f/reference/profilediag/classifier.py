def classify_error(log_text, environment_info):
    text_lower = log_text.lower()
    in_container = environment_info.get("in_container", True)
    has_sys_admin = environment_info.get("has_sys_admin", False)
    perf_paranoid = environment_info.get("perf_event_paranoid", 2)

    if "perf_event_open" in text_lower or "permission denied" in text_lower or "paranoid" in text_lower:
        if in_container and not has_sys_admin:
            return "CONTAINER_MISSING_SYS_ADMIN"
        if perf_paranoid > 1:
            return "BARE_METAL_PERF_PARANOID"
        return "BARE_METAL_PERMISSION_DENIED"

    if "nvml" in text_lower or "driver" in text_lower or "initialization error" in text_lower:
        return "DRIVER_OR_NVML_FAILURE"

    if "device" in text_lower and ("not found" in text_lower or "unavailable" in text_lower):
        return "DEVICE_UNAVAILABLE"

    if in_container and not has_sys_admin and ("sys_admin" in text_lower or "capability" in text_lower or "ptrace" in text_lower):
        return "CONTAINER_MISSING_SYS_ADMIN"

    return "UNKNOWN_PROFILING_FAILURE"
