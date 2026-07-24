import numpy as np


def _encode_decode_value(v, fmt):
    if fmt == "e4m3":
        eb, mb, bias = 4, 3, 7
    else:
        eb, mb, bias = 5, 2, 15

    bits = np.frombuffer(np.float32(v).tobytes(), dtype=np.uint32)[0]
    sign = int(bits >> 31)
    exp = int((bits >> 23) & 0xff)
    frac = int(bits & 0x7fffff)

    if exp == 0:
        return 0.0

    unbiased = exp - 127
    fp_exp = unbiased + bias
    max_exp = (1 << eb) - 2

    if fp_exp <= 0:
        fp_exp = 0
        mant = 0
    elif fp_exp > max_exp:
        fp_exp = max_exp
        mant = (1 << mb) - 1
    else:
        shift = 23 - mb
        mant = frac >> shift
        rem = frac & ((1 << shift) - 1)
        halfway = 1 << (shift - 1)
        if rem > halfway or (rem == halfway and (mant & 1)):
            mant += 1
            if mant == (1 << mb):
                mant = 0
                fp_exp += 1
                if fp_exp > max_exp:
                    fp_exp = max_exp
                    mant = (1 << mb) - 1

    code = (sign << (eb + mb)) | (fp_exp << mb) | mant

    out_sign = -1.0 if (code >> (eb + mb)) else 1.0
    out_exp = (code >> mb) & ((1 << eb) - 1)
    out_mant = code & ((1 << mb) - 1)

    if out_exp == 0:
        return 0.0

    return out_sign * (2.0 ** (out_exp - bias)) * (
        1.0 + out_mant / (2.0 ** mb)
    )


def _oracle(x, fmt):
    return np.array([_encode_decode_value(v, fmt) for v in x.ravel()], dtype=np.float64).reshape(x.shape)


def grade(sol, fx) -> dict:
    cases = [
        np.array([0.0, 1.0, -1.0, 1.5, -2.25, 3.75], dtype=np.float32),
        np.array([1e-5, -1e-3, 12.5, 1000.0], dtype=np.float32),
        np.array([[0.25, -0.75], [16.0, -64.0]], dtype=np.float32),
    ]

    worst = 0.0
    try:
        for arr in cases:
            for fmt in ("e4m3", "e5m2"):
                got = np.asarray(sol.fp8_roundtrip(arr, fmt), dtype=np.float64)
                ref = _oracle(arr, fmt)
                worst = max(worst, float(np.max(np.abs(got - ref))))
    except Exception:
        worst = float("inf")

    return {"max_abs_err": worst}
