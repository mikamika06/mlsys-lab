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

    def addr_a(i, k):
        return base_a + (i * N + k) * elem

    def addr_b(k, j):
        return base_b + (k * N + j) * elem

    def addr_c(i, j):
        return base_c + (i * N + j) * elem

    addrs = []
    ranges = range(N)

    if order == "ijk":
        for i in ranges:
            for j in ranges:
                for k in ranges:
                    addrs.append(addr_a(i, k))
                    addrs.append(addr_b(k, j))
                    addrs.append(addr_c(i, j))
    elif order == "ikj":
        for i in ranges:
            for k in ranges:
                for j in ranges:
                    addrs.append(addr_a(i, k))
                    addrs.append(addr_b(k, j))
                    addrs.append(addr_c(i, j))
    elif order == "jki":
        for j in ranges:
            for k in ranges:
                for i in ranges:
                    addrs.append(addr_a(i, k))
                    addrs.append(addr_b(k, j))
                    addrs.append(addr_c(i, j))
    else:
        raise ValueError(order)

    return addrs


def _miss_count(addrs):
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


def _reference():
    orders = ["ijk", "ikj", "jki"]
    scored = [(order, _miss_count(_trace(order))) for order in orders]
    scored.sort(key=lambda x: (x[1], x[0]))
    return [x[0] for x in scored]


def grade(sol, fx) -> dict:
    expected = _reference()
    try:
        got = sol.rank_matmul_orders()
    except Exception:
        return {"exact_match": 0.0}

    if got != expected:
        return {"exact_match": 0.0}
    return {"exact_match": 1.0}
