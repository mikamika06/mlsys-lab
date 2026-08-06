import math

def log1p_tiny(x: list[float]) -> list[float]:
    """
    Accurate computation of log(1+x) for tiny x.
    Uses math.log1p which is stable for |x| << 1.
    """
    out = []
    for val in x:
        out.append(math.log1p(val))
    return out
