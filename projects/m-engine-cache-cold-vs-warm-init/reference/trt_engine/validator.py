import hashlib


def validate_cache_fingerprint(engine_bytes, expected_hash):
    if not engine_bytes:
        return False
    h = hashlib.sha256(engine_bytes).hexdigest()
    return h == expected_hash


def detect_invalidation_trigger(cached_profile, current_profile):
    if not cached_profile or not current_profile:
        return True
    for key in ("min", "opt", "max"):
        if cached_profile.get(key) != current_profile.get(key):
            return True
    return False
