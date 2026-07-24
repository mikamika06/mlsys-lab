from decimal import Decimal, getcontext

def _reference(a: float, b: float) -> bool:
    """Compute whether a - b suffers catastrophic cancellation using high precision."""
    getcontext().prec = 50
    exact = Decimal(str(a)) - Decimal(str(b))
    float_diff = a - b
    if exact == 0:
        return float_diff != 0.0
    rel_err = abs(float_diff - float(exact)) / abs(float(exact))
    return rel_err > 1e-12

def grade(sol, fx) -> dict:
    cases = [
        (1e16 + 1, 1e16),          # catastrophic cancellation
        (1000.0, 999.9999),        # small relative error
        (1.234567890123456, 1.234567890123455),  # negligible difference
        (1.0, 1.0),                # exact zero
        (1e-10 + 1e-20, 1e-10)     # tiny difference relative to magnitude
    ]
    ok = 1.0
    for a, b in cases:
        try:
            got = sol.spot_cancellation(a, b)
        except Exception:
            return {"exact_match": 0.0}
        expected = _reference(a, b)
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
