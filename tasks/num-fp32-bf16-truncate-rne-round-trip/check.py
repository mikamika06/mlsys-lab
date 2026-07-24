import numpy as np
from mlsys import scorers


def grade(sol, fx) -> dict:
    values = np.array(
        [
            0.0,
            1.0,
            -1.0,
            1.5,
            -2.5,
            3.1415927,
            1.00390625,
            np.float32(1.0 + 2 ** -8),
            np.float32(np.nextafter(1.0, np.float32(2.0))),
            np.float32(65504.0),
            np.float32(-65504.0),
            np.float32(np.inf),
            np.float32(-np.inf),
            np.float32(np.nan),
        ],
        dtype=np.float32,
    )

    values = np.concatenate(
        [
            values,
            np.linspace(-1000, 1000, 257, dtype=np.float32),
        ]
    )

    try:
        import ml_dtypes
        ref = np.asarray(values, dtype=ml_dtypes.bfloat16).view(np.uint16)
    except Exception:
        return {"byte_exact_fraction": 0.0}

    try:
        got = sol.fp32_to_bf16_codes(values)
        got = np.asarray(got)
    except Exception:
        return {"byte_exact_fraction": 0.0}

    if got.shape != ref.shape or got.dtype != np.uint16:
        return {"byte_exact_fraction": 0.0}

    return {
        "byte_exact_fraction": scorers.byte_exact_fraction(got, ref)
    }
