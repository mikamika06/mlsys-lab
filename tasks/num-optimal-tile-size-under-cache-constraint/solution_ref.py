def _traffic(m, n, k, b):
    r = (m + b - 1) // b
    c = (n + b - 1) // b
    t = (k + b - 1) // b
    return r * c * t * b * b * 2


def optimal_tile_size(m, n, k, cache_bytes, element_bytes):
    max_b = int((cache_bytes // (3 * element_bytes)) ** 0.5)
    best_b = 1
    best_cost = None

    for b in range(1, max_b + 1):
        cost = _traffic(m, n, k, b)
        if best_cost is None or cost < best_cost or (cost == best_cost and b > best_b):
            best_cost = cost
            best_b = b

    return best_b
