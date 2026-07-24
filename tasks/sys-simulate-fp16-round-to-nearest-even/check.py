import inspect
import numpy as np


def grade(sol, fx) -> dict:
    cases = [
        np.array(
            [
                0.0,
                -0.0,
                1.0,
                -1.0,
                1.00048828125,
                65504.0,
                70000.0,
                1e-8,
                -1e-8,
                np.inf,
                -np.inf,
                np.nan,
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                np.float32(i) * np.float32(0.001)
                for i in range(-1000, 1001)
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                np.float32(2.0 ** e)
                for e in range(-30, 16)
            ],
            dtype=np.float32,
        ),
    ]

    try:
        source = inspect.getsource(sol.fp32_to_fp16_rne)
    except Exception:
        return {"max_abs_err": 1.0}

    if "astype" in source or "float16" in source and "astype" in source:
        return {"max_abs_err": 1.0}

    worst = 0.0
    for x in cases:
        try:
            got = sol.fp32_to_fp16_rne(x)
        except Exception:
            return {"max_abs_err": 1.0}

        if not isinstance(got, np.ndarray) or got.dtype != np.float16:
            return {"max_abs_err": 1.0}

        ref = x.astype(np.float16)
        a = got.astype(np.float32)
        b = ref.astype(np.float32)

        diff = np.abs(a - b)
        finite = np.isfinite(diff)
        if np.any(finite):
            worst = max(worst, float(np.max(diff[finite])))

        if np.any(np.isnan(a) != np.isnan(b)):
            return {"max_abs_err": 1.0}

        if np.any(np.isinf(a) != np.isinf(b)):
            return {"max_abs_err": 1.0}

    return {"max_abs_err": worst}
