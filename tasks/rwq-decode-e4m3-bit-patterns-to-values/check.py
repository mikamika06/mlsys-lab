import numpy as np

def _e4m3_oracle():
    """Build the canonical 256-entry E4M3 decoding via vectorised NumPy.

    This is the ground-truth reference; check.py never hardcodes expected values.
    """
    codes = np.arange(256, dtype=np.uint8)

    sign_bit = (codes >> 7).astype(np.uint8)
    exp_bits = ((codes >> 3) & 0x0F).astype(np.uint16)
    man_bits = (codes & 0x07).astype(np.float64)

    sign = np.where(sign_bit == 0, 1.0, -1.0)

    result = np.zeros(256, dtype=np.float64)

    # Normalized numbers: exponent != 0
    norm = exp_bits != 0
    result[norm] = sign[norm] * np.ldexp(
        1.0 + man_bits[norm] / 8.0,
        (exp_bits[norm].astype(np.int16) - 7).astype(np.int16),
    )

    # Subnormals: exponent == 0, mantissa != 0
    sub = (exp_bits == 0) & (man_bits != 0.0)
    result[sub] = sign[sub] * np.ldexp(man_bits[sub] / 8.0, -6)

    # Zeros: exponent == 0, mantissa == 0 (result stays 0.0 with correct sign)
    zero = (exp_bits == 0) & (man_bits == 0.0)
    result[zero] = np.where(sign_bit[zero] == 0, 0.0, -0.0)

    return result

def grade(sol, fx) -> dict:
    """Grade decode_e4m3 against the vectorised E4M3 oracle."""
    ref = _e4m3_oracle()

    try:
        codes = np.arange(256, dtype=np.uint8)
        got = np.asarray(sol.decode_e4m3(codes), dtype=np.float64)
    except Exception:
        return {"exact_match": 0.0}

    if got.shape != ref.shape:
        return {"exact_match": 0.0}

    # NaN == NaN is True in np.array_equal, so this handles all NaN cases.
    ok = 1.0 if np.array_equal(got, ref) else 0.0
    return {"exact_match": ok}
