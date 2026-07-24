from collections import OrderedDict


def _misses(M, N, K, cache_lines, line_elems, tile):
    tm, tn, tk = tile
    cache = OrderedDict()
    misses = 0

    def access(a, index):
        nonlocal misses
        key = (a, index // line_elems)
        if key in cache:
            cache.move_to_end(key)
        else:
            misses += 1
            cache[key] = None
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


def choose_tile_sizes(M, N, K, cache_lines, line_elems, candidates):
    best = candidates[0]
    best_misses = _misses(M, N, K, cache_lines, line_elems, best)
    for tile in candidates[1:]:
        miss = _misses(M, N, K, cache_lines, line_elems, tile)
        if miss < best_misses:
            best = tile
            best_misses = miss
    return tuple(best)
