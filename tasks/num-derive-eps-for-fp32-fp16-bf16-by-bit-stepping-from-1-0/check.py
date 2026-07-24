import numpy as np
from mlsys import scorers

def _reference_eps() -> np.ndarray:
    eps32 = np.nextafter(np.float32(1.0), np.float32(np.inf)) - np.float32(1.0)
    eps16 = np.nextafter(np.float16(1.0), np.float16(np.inf)) - np.float16(1.0)

    # Compute BF16 epsilon via bit manipulation to avoid dependency on np.bfloat16
    bits = np.uint32(0x3f800000)          # float32 representation of 1.0
    next_bits = bits + (1 << 16)          # increment mantissa in the top 16 bits
    epsbf16 = np.float32(next_bits.view(np.float32)) - 1.0

    return np.array([eps32, eps16, epsbf16], dtype=np.float64)

def grade(sol, fx) -> dict:
    try:
        got = sol.derive_eps()
    except Exception:
        return {"rel_err": 1.0}
    ref = _reference_eps()

    # Ensure we compare as float64 arrays
    try:
        got_arr = np.asarray(got, dtype=np.float64)
    except Exception:
        return {"rel_err": 1.0}

    if got_arr.shape != (3,):
        return {"rel_err": 1.0}

    err = scorers.rel_err(ref, got_arr)
    return {"rel_err": float(err)}
