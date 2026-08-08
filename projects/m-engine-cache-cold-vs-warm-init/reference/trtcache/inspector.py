def check_cache_validity(config, cached_meta):
    reasons = []
    if config.get("trt_version") != cached_meta.get("trt_version"):
        reasons.append("trt_version_mismatch")
    if config.get("shape_profile") != cached_meta.get("shape_profile"):
        reasons.append("shape_profile_mismatch")
    if config.get("ep_options_hash") != cached_meta.get("ep_options_hash"):
        reasons.append("ep_options_hash_mismatch")
    if config.get("model_hash") != cached_meta.get("model_hash"):
        reasons.append("model_hash_mismatch")

    valid = len(reasons) == 0
    return {"valid": valid, "reasons": reasons}
