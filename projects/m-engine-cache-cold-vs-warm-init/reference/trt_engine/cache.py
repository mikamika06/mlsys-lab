import hashlib


def verify_engine_cache(cache_meta, runtime_config):
    if not cache_meta or not isinstance(cache_meta, dict):
        return False
    if cache_meta.get("device_id") != runtime_config.get("device_id"):
        return False
    if cache_meta.get("trt_version") != runtime_config.get("trt_version"):
        return False
    profiles = cache_meta.get("profiles", [])
    curr_profiles = runtime_config.get("profiles", [])
    if len(profiles) != len(curr_profiles):
        return False
    for p, c in zip(profiles, curr_profiles):
        if p.get("min") != c.get("min") or p.get("max") != c.get("max"):
            return False
    return True


def compute_warm_init_latency(latencies):
    if not latencies:
        return 0.0
    return float(sum(latencies)) / float(len(latencies))
