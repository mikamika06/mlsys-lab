def validate_cache(meta, env):
    if not meta or not env:
        return "cold"
    if meta.get("version") != env.get("version"):
        return "invalid_version"
    if meta.get("device") != env.get("device"):
        return "invalid_device"
    return "warm"
