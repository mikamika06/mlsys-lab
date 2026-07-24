def retained_kv_indices(n: int, s: int, w: int) -> list[int]:
    sink_end = min(s, n)
    window_start = max(0, n - w)

    retained = set(range(sink_end))
    retained.update(range(window_start, n))
    return sorted(retained)
