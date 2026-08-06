import math

def _ref(data, mask):
    """Oracle: sum via filtered elements — never forms 0*inf."""
    total = 0.0
    for d, m in zip(data, mask):
        if m:
            total += d
    return float(total)

def grade(sol, fx) -> dict:
    cases = [
        # basic finite
        ([1.0, 2.0, 3.0], [True, False, True]),
        # 0*inf trap: inf masked out
        ([1.0, float("inf"), 3.0], [True, False, True]),
        # NaN masked in should propagate
        ([float("nan"), 2.0, float("inf")], [True, True, False]),
        # -inf masked in
        ([float("inf"), float("-inf"), 0.0], [False, True, True]),
        # both inf and nan masked out
        ([float("inf"), float("nan"), 1.0], [False, False, True]),
        # empty reduction
        ([0.0, 0.0, 0.0], [False, False, False]),
        # large negative inf
        ([float("-inf"), 5.0], [True, True]),
        # double NaN masked in
        ([float("nan"), float("nan")], [True, True]),
    ]

    ok = 1.0
    for data, mask in cases:
        try:
            got = sol.masked_sum(list(data), list(mask))
        except Exception:
            ok = 0.0
            break

        expected = _ref(data, mask)

        # NaN-safe comparison
        if math.isnan(expected):
            if not math.isnan(got):
                ok = 0.0
                break
        elif got != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
