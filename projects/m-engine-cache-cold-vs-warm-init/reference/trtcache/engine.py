def classify_init_state(engine_meta, cache_store):
    key = engine_meta.get("hash")
    if not key or key not in cache_store:
        return "cold"
    stored = cache_store[key]
    if stored.get("profile_signature") != engine_meta.get("profile_signature"):
        return "invalidated"
    if stored.get("plugin_version") != engine_meta.get("plugin_version"):
        return "invalidated"
    return "warm"


def verify_cache_validity(engine_meta, current_profile):
    if engine_meta.get("profile_signature") != current_profile:
        return False
    return engine_meta.get("is_valid", True)
