from collections import OrderedDict


def modeled_access_count(n, cache_bytes, line_bytes, tile_sizes):
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
                    for i in range(ii, min(ii + tile, n)):
                        for k in range(kk, min(kk + tile, n)):
                            access(base_a + (i * n + k) * elem)
                            for j in range(jj, min(jj + tile, n)):
                                access(base_b + (k * n + j) * elem)
                                access(base_c + (i * n + j) * elem)
        return misses

    return {int(t): simulate(int(t)) for t in tile_sizes}
