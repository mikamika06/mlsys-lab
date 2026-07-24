import numpy as np


def _oracle_unpack_nf4(packed, absmax):
    levels = np.array(
        [
            -1.0000, -0.6962, -0.5251, -0.3949,
            -0.2844, -0.1848, -0.0911, 0.0,
             0.0796,  0.1609,  0.2461,  0.3379,
             0.4407,  0.5626,  0.7229, 1.0000,
        ],
        dtype=np.float64,
    )
    packed = np.asarray(packed, dtype=np.uint8)
    high = packed >> np.uint8(4)
    low = packed & np.uint8(15)
    codes = np.empty(packed.size * 2, dtype=np.uint8)
    codes[0::2] = high
    codes[1::2] = low
    weights = levels[codes.astype(np.int64)] * float(absmax)
    return codes, weights


def grade(sol, fx) -> dict:
    cases = [
        (np.array([0x01, 0xFE, 0x87], dtype=np.uint8), 2.0),
        (np.array([0x00, 0xFF, 0x78, 0x89], dtype=np.uint8), 0.5),
        (np.array([0xAB, 0xCD, 0xEF], dtype=np.uint8), 3.25),
    ]

    code_score = 1.0
    max_error = 0.0

    for packed, scale in cases:
        ref_codes, ref_weights = _oracle_unpack_nf4(packed, scale)
        try:
            got_codes, got_weights = sol.unpack_nf4(packed, scale)
            got_codes = np.asarray(got_codes)
            got_weights = np.asarray(got_weights)
        except Exception:
            return {"code_exact": 0.0, "max_abs_err": float("inf")}

        if not np.array_equal(got_codes, ref_codes):
            code_score = 0.0

        if got_weights.shape != ref_weights.shape:
            max_error = float("inf")
        else:
            max_error = max(
                max_error,
                float(np.max(np.abs(got_weights.astype(np.float64) - ref_weights))),
            )

    return {
        "code_exact": code_score,
        "max_abs_err": max_error,
    }
