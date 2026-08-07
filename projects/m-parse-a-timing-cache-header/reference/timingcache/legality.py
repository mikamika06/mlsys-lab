def is_cache_reusable(cache_header: dict, target_env: dict, required_tactic_sources: int) -> bool:
    if cache_header["sm_version"] != target_env["sm_version"]:
        return False

    if cache_header["sm_count"] != target_env["sm_count"]:
        return False

    c_trt = cache_header["trt_version"]
    t_trt = target_env["trt_version"]
    if (c_trt[0], c_trt[1]) != (t_trt[0], t_trt[1]):
        return False

    if (cache_header["tactic_sources"] & required_tactic_sources) != required_tactic_sources:
        return False

    return True
