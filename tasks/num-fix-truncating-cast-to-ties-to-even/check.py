import numpy as np
from mlsys import scorers


def grade(sol, fx) -> dict:
    values = np.array(
        [
            1.0,
            1.0000000596046448,
            1.0000001788139343,
            -1.0000000596046448,
            0.3333333333333333,
            1e-40,
            -1e-40,
            np.float64(np.nextafter(np.float32(1.0), np.float32(2.0))),
            np.float64(np.nextafter(np.float32(-1.0), np.float32(-2.0))),
        ],
        dtype=np.float64,
    )

    oracle = values.astype(np.float32)

    try:
        got = sol.cast_f32_rne(values)
    except Exception:
        return {"byte_exact_fraction": 0.0}

    try:
        score = scorers.byte_exact_fraction(oracle, got)
    except Exception:
        score = 0.0

    return {"byte_exact_fraction": score}
