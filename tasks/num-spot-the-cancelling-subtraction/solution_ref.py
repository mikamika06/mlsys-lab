from decimal import Decimal, getcontext

def spot_cancellation(a: float, b: float) -> bool:
    """
    Return True if the subtraction a - b suffers from catastrophic cancellation.
    Uses high‑precision arithmetic to compute the exact difference and compares
    it with the IEEE‑754 double result.
    """
    # Use decimal for arbitrary precision; convert via string to preserve value
    getcontext().prec = 50
    exact = Decimal(str(a)) - Decimal(str(b))
    float_diff = a - b
    if exact == 0:
        return float_diff != 0.0
    rel_err = abs(float_diff - float(exact)) / abs(float(exact))
    return rel_err > 1e-12
