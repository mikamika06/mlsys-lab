import numpy as np

def _reference_boundary() -> float:
    """
    Compute the smallest positive real number that overflows to inf in FP16
    but stays finite in BF16 (or, if unavailable, in FP32).  Uses a binary
    search between the known maximum finite value (65504) and an upper bound.
    """
    low = 65504.0
    high = 70000.0

    # Choose the surrogate for BF16: prefer np.bfloat16 if present.
    try:
        cast_bf = np.bfloat16
    except AttributeError:
        cast_bf = np.float32

    while high - low > 1e-6:
        mid = (low + high) / 2.0
        # Check overflow in FP16 and finiteness in the surrogate.
        if np.float16(mid) == np.inf and cast_bf(mid) != np.inf:
            high = mid
        else:
            low = mid

    return float(high)

def grade(sol, fx) -> dict:
    """
    Grade the candidate solution.

    Parameters
    ----------
    sol : module
        The student's module containing `find_fp16_overflow_boundary`.
    fx : any
        Unused; present for API compatibility.

    Returns
    -------
    dict
        {"exact_match": 1.0} if the returned value matches the reference,
        otherwise {"exact_match": 0.0}.
    """
    try:
        val = sol.find_fp16_overflow_boundary()
    except Exception:
        return {"exact_match": 0.0}

    # Ensure we have a scalar float
    try:
        val_float = float(val)
    except Exception:
        return {"exact_match": 0.0}

    ref = _reference_boundary()

    ok = 1.0 if val_float == ref else 0.0
    return {"exact_match": ok}
