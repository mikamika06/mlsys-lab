def classify_engine(header: dict, runtime_env: dict) -> dict:
    """Classify engine compatibility against host runtime environment."""
    if not header.get("valid", False):
        return {"status": "CORRUPTED_HEADER", "penalty": None}
    if header.get("platform") != runtime_env.get("platform"):
        return {"status": "PLATFORM_MISMATCH", "penalty": None}

    h_ver = header.get("trt_version", (0, 0, 0, 0))
    r_ver = runtime_env.get("trt_version", (0, 0, 0, 0))
    if h_ver[0] != r_ver[0] or h_ver > r_ver:
        return {"status": "VERSION_MISMATCH", "penalty": None}

    h_sm = header.get("sm_arch", 0)
    r_sm = runtime_env.get("sm_arch", 0)
    if h_sm == r_sm:
        return {"status": "OK", "penalty": 1.0}

    if header.get("hardware_compatible", False) and r_sm > h_sm:
        penalty = round(1.0 + 0.05 * (r_sm - h_sm), 4)
        return {"status": "OK", "penalty": penalty}

    return {"status": "INCOMPATIBLE_HARDWARE", "penalty": None}
