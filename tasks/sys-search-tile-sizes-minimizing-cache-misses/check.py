from collections import OrderedDict


def _simulate(M, N, K, cache_lines, line_elems, tile):
    tm, tn, tk = tile
    cache = OrderedDict()
    misses = 0

    def access(array_id, index):
        nonlocal misses
        line = (array_id, index // line_elems)
        if line in cache:
            cache.move_to_end(line)
        else:
            misses += 1
            cache[line] = None
            if len(cache) > cache_lines:
                cache.popitem(last=False)

    for i0 in range(0, M, tm):
        for j0 in range(0, N, tn):
            for k0 in range(0, K, tk):
                for i in range(i0, min(i0 + tm, M)):
                    for k in range(k0, min(k0 + tk, K)):
                        access(0, i * K + k)
                for k in range(k0, min(k0 + tk, K)):
                    for j in range(j0, min(j0 + tn, N)):
                        access(1, k * N + j)
                for i in range(i0, min(i0 + tm, M)):
                    for j in range(j0, min(j0 + tn, N)):
                        access(2, i * N + j)
    return misses


def _oracle(M, N, K, cache_lines, line_elems, candidates):
    best = None
    for tile in candidates:
        miss = _simulate(M, N, K, cache_lines, line_elems, tile)
        if best is None or miss < best:
            best = miss
    return best


def grade(sol, fx) -> dict:
    cases = [
        (16, 16, 16, 8, 4, [(2, 2, 2), (4, 4, 4), (8, 8, 2)]),
        (31, 17, 23, 16, 8, [(1, 4, 4), (4, 4, 8), (8, 2, 2), (16, 8, 4)]),
        (24, 40, 12, 12, 4, [(2, 8, 2), (8, 8, 4), (12, 4, 6)]),
    ]
    ratios = []
    for M, N, K, cache_lines, line_elems, candidates in cases:
        try:
            got = sol.choose_tile_sizes(
                M, N, K, cache_lines, line_elems, list(candidates)
            )
        except Exception:
            return {"modeled_mem_access": float("inf")}

        if tuple(got) not in candidates:
            return {"modeled_mem_access": float("inf")}

        actual = _simulate(M, N, K, cache_lines, line_elems, tuple(got))
        optimum = _oracle(M, N, K, cache_lines, line_elems, candidates)
        ratios.append(actual / max(optimum, 1))

    return {"modeled_mem_access": max(ratios)}
