def resolve_default_context(profile):
    if profile["vram_bytes"] <= 8 * 1024**3:
        return 4096
    return 32768
