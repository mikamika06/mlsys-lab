def recommend_wired_limit(memsize: int) -> int:
    gb = 1024 ** 3
    if memsize <= 16 * gb:
        os_buffer = 3 * gb
    elif memsize <= 32 * gb:
        os_buffer = 6 * gb
    else:
        os_buffer = 8 * gb

    limit_by_buffer = memsize - os_buffer
    limit_by_ratio = int(memsize * 0.85)
    return min(limit_by_buffer, limit_by_ratio)


def tune_limits(memsize: int, model_bytes: int) -> tuple[int, int]:
    wired = recommend_wired_limit(memsize)
    buffer = 256 * 1024 * 1024
    cache = wired - model_bytes - buffer
    return wired, max(0, cache)
