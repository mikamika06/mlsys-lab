import math


def exp_minus_one(x: list[float]) -> list[float]:
    """Accurate e**x - 1, including for |x| far below the fp64 epsilon."""
    out = []

    for xi in x:
        try:
            u = math.exp(xi)
        except OverflowError:
            u = float("inf")

        d = u - 1.0

        if u == 0.0:
            val = -1.0
        elif d == 0.0:
            val = xi
        else:
            try:
                lu = math.log(u)
            except ValueError:
                lu = float("nan")
            val = d * (xi / lu)

        out.append(val)

    return out
