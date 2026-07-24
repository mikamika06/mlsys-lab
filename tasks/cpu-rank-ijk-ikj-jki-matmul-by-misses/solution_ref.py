from mlsys.sim import cache as cachesim


N = 24
LINE_BYTES = 64
SETS = 32
WAYS = 2


def _trace(order):
    elem = 8
    base_a = 0
    base_b = N * N * elem
    base_c = 2 * N * N * elem

    def a(i, k):
        return base_a + (i * N + k) * elem

    def b(k, j):
        return base_b + (k * N + j) * elem

    def c(i, j):
        return base_c + (i * N + j) * elem

    out = []

    if order == "ijk":
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    out.extend((a(i, k), b(k, j), c(i, j)))
    elif order == "ikj":
        for i in range(N):
            for k in range(N):
                for j in range(N):
                    out.extend((a(i, k), b(k, j), c(i, j)))
    else:
        for j in range(N):
            for k in range(N):
                for i in range(N):
                    out.extend((a(i, k), b(k, j), c(i, j)))

    return out


def _misses(addrs):
    result = cachesim.simulate(
        addrs,
        line_bytes=LINE_BYTES,
        sets=SETS,
        ways=WAYS,
    )
    if isinstance(result, dict):
        return result["misses"]
    if hasattr(result, "misses"):
        return result.misses
    return result[1]


def rank_matmul_orders():
    scores = []
    for order in ["ijk", "ikj", "jki"]:
        scores.append((order, _misses(_trace(order))))
    scores.sort(key=lambda x: (x[1], x[0]))
    return [x[0] for x in scores]
