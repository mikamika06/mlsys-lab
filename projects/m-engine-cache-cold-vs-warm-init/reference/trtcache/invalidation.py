def check_cache_validity(cache_meta, current_meta):
    if cache_meta.get("trt_version") != current_meta.get("trt_version"):
        return False
    if cache_meta.get("gpu_arch") != current_meta.get("gpu_arch"):
        return False
    if cache_meta.get("profile_hash") != current_meta.get("profile_hash"):
        return False
    if cache_meta.get("shape_signature") != current_meta.get("shape_signature"):
        return False
    return True
