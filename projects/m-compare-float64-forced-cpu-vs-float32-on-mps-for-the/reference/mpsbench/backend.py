def inspect_backend(info_dict):
    built = bool(info_dict.get("is_built", False))
    avail = bool(info_dict.get("is_available", False))
    if not built:
        status = "unsupported_build"
    elif not avail:
        status = "built_but_unavailable"
    else:
        status = "mps_available"
    return {
        "is_built": built,
        "is_available": avail,
        "status": status,
        "can_use_mps": built and avail,
    }


def resolve_device_dtype(requested_dtype, backend_info):
    insp = inspect_backend(backend_info)
    if requested_dtype == "float64":
        return {"device": "cpu", "dtype": "float64", "forced_cpu": True}
    if insp["can_use_mps"] and requested_dtype == "float32":
        return {"device": "mps", "dtype": "float32", "forced_cpu": False}
    return {"device": "cpu", "dtype": requested_dtype, "forced_cpu": True}
