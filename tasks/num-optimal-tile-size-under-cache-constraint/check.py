def _traffic(m, n, k, b):
    r = (m + b - 1) // b
    c = (n + b - 1) // b
    t = (k + b - 1) // b
    return r * c * t * b * b * 2


def _oracle(m, n, k, cache_bytes, element_bytes):
    max_b = int((cache_bytes // (3 * element_bytes)) ** 0.5)
    best_b = 0
    best = None
    for b in range(1, max_b + 1):
        value = _traffic(m, n, k, b)
        if best is None or value < best or (value == best and b > best_b):
            best = value
            best_b = b
    return best_b, best


def grade(sol, fx) -> dict:
    cases = [
        (1024, 1024, 1024, 196608, 8),
        (513, 257, 129, 32768, 4),
        (4096, 1024, 2048, 262144, 8),
        (100, 1000, 300, 65536, 8),
        (777, 333, 555, 131072, 4),
    ]

    ratios = []
    for m, n, k, cache_bytes, element_bytes in cases:
        ref_b, ref_cost = _oracle(m, n, k, cache_bytes, element_bytes)
        try:
            got_b = sol.optimal_tile_size(m, n, k, cache_bytes, element_bytes)
            got_cost = _traffic(m, n, k, got_b)
        except Exception:
            return {"modeled_access_count": float("inf")}

        if not isinstance(got_b, int):
            return {"modeled_access_count": float("inf")}
        if got_b < 1 or 3 * got_b * got_b * element_bytes > cache_bytes:
            return {"modeled_access_count": float("inf")}

        ratios.append(got_cost / ref_cost)

    return {"modeled_access_count": max(ratios)}
