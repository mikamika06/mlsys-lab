from decimal import Decimal, getcontext
import math

getcontext().prec = 100


def _oracle(a, b, c):
    da = Decimal(str(a))
    db = Decimal(str(b))
    dc = Decimal(str(c))
    disc = db * db - Decimal(4) * da * dc
    sqrt_disc = disc.sqrt()
    if db >= 0:
        q = -(db + sqrt_disc) / Decimal(2)
    else:
        q = -(db - sqrt_disc) / Decimal(2)
    x1 = q / da
    x2 = dc / q
    return float(x1), float(x2)


def _pair_error(got, ref):
    g = [float(got[0]), float(got[1])]
    r = [float(ref[0]), float(ref[1])]
    direct = max(
        abs(g[0] - r[0]) / (abs(r[0]) + 1e-300),
        abs(g[1] - r[1]) / (abs(r[1]) + 1e-300),
    )
    swapped = max(
        abs(g[0] - r[1]) / (abs(r[1]) + 1e-300),
        abs(g[1] - r[0]) / (abs(r[0]) + 1e-300),
    )
    return min(direct, swapped)


def grade(sol, fx) -> dict:
    cases = [
        (1.0, 1e8, 1.0),
        (3.0, -4e10, 2.0),
        (1.0, 1e12, 1.0),
        (7.0, -9e9, 0.25),
        (0.5, 5e7, 3.0),
    ]
    worst = 0.0
    try:
        for a, b, c in cases:
            ref = _oracle(a, b, c)
            got = sol.solve_quadratic(a, b, c)
            err = _pair_error(got, ref)
            worst = max(worst, err)
    except Exception:
        worst = float("inf")
    return {"rel_err": worst}
