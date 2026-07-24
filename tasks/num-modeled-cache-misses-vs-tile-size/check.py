from collections import OrderedDict


def _reference(n, cache_bytes, line_bytes, tile_sizes):
    def simulate(tile):
        capacity = cache_bytes // line_bytes
        cache = OrderedDict()
        misses = 0
        elem = 8
        base_a = 0
        base_b = n * n * elem
        base_c = 2 * n * n * elem

        def access(addr):
            nonlocal misses
            line = addr // line_bytes
            if line in cache:
                cache.move_to_end(line)
            else:
                misses += 1
                cache[line] = None
                cache.move_to_end(line)
                if len(cache) > capacity:
                    cache.popitem(last=False)

        for ii in range(0, n, tile):
            for kk in range(0, n, tile):
                for jj in range(0, n, tile):
                    i_end = min(ii + tile, n)
                    k_end = min(kk + tile, n)
                    j_end = min(jj + tile, n)
                    for i in range(ii, i_end):
                        for k in range(kk, k_end):
                            access(base_a + (i * n + k) * elem)
                            for j in range(jj, j_end):
                                access(base_b + (k * n + j) * elem)
                                access(base_c + (i * n + j) * elem)
        return misses

    return {int(t): simulate(int(t)) for t in tile_sizes}


def grade(sol, fx) -> dict:
    cases = [
        (4, 128, 64, [1, 2, 4]),
        (5, 256, 64, [2, 3, 5]),
        (8, 512, 128, [2, 4, 8]),
        (7, 96, 32, [1, 4, 7]),
    ]
    ok = 1.0
    for args in cases:
        try:
            got = sol.modeled_access_count(*args)
        except Exception:
            ok = 0.0
            break
        if got != _reference(*args):
            ok = 0.0
            break
    return {"modeled_access_count": ok}
