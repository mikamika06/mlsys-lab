import math


def _oracle_cost(candidate, shape):
    m, n, k = shape
    tm = int(candidate["tile_m"])
    tn = int(candidate["tile_n"])
    tk = int(candidate["tile_k"])
    tiles = (
        math.ceil(m / tm)
        * math.ceil(n / tn)
        * math.ceil(k / tk)
    )
    return tiles * tm * tn * tk


def _reference(candidates, shape):
    return [
        x["id"]
        for x in sorted(
            candidates,
            key=lambda c: (_oracle_cost(c, shape), c["id"]),
        )
    ]


def grade(sol, fx) -> dict:
    cases = [
        (
            [
                {"id": "small", "tile_m": 8, "tile_n": 8, "tile_k": 8},
                {"id": "wide", "tile_m": 32, "tile_n": 8, "tile_k": 8},
                {"id": "balanced", "tile_m": 16, "tile_n": 16, "tile_k": 16},
            ],
            (64, 64, 64),
        ),
        (
            [
                {"id": "c", "tile_m": 7, "tile_n": 9, "tile_k": 5},
                {"id": "a", "tile_m": 7, "tile_n": 9, "tile_k": 5},
                {"id": "b", "tile_m": 16, "tile_n": 4, "tile_k": 8},
            ],
            (31, 27, 19),
        ),
        (
            [
                {"id": "large", "tile_m": 64, "tile_n": 64, "tile_k": 64},
                {"id": "tiny", "tile_m": 4, "tile_n": 4, "tile_k": 4},
                {"id": "mid", "tile_m": 16, "tile_n": 8, "tile_k": 8},
            ],
            (128, 96, 80),
        ),
    ]

    ok = 1.0
    for candidates, shape in cases:
        try:
            got = sol.rank_schedules(list(candidates), shape)
        except Exception:
            ok = 0.0
            break
        if got != _reference(candidates, shape):
            ok = 0.0
            break
    return {"exact_match": ok}
