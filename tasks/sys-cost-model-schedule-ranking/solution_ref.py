def rank_schedules(candidates, shape):
    import math

    m, n, k = shape

    def cost(c):
        tm = int(c["tile_m"])
        tn = int(c["tile_n"])
        tk = int(c["tile_k"])
        return (
            math.ceil(m / tm)
            * math.ceil(n / tn)
            * math.ceil(k / tk)
            * tm
            * tn
            * tk
        )

    return [
        c["id"]
        for c in sorted(candidates, key=lambda x: (cost(x), x["id"]))
    ]
