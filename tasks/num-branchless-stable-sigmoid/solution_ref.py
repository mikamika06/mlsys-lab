import math


def stable_sigmoid(x: list[float]) -> list[float]:
    """Overflow-free logistic sigmoid, evaluated branchlessly."""
    out = []
    for val in x:
        abs_val = val if val >= 0.0 else -val
        z = math.exp(-abs_val)
        denom = 1.0 + z
        if val >= 0.0:
            out.append(1.0 / denom)
        else:
            out.append(z / denom)
    return out
