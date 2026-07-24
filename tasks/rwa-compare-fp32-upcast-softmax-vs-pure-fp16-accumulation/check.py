import numpy as np


def _oracle(x):
    y = np.asarray(x, dtype=np.float64)
    y = y - np.max(y, axis=1, keepdims=True)
    e = np.exp(y)
    return e / np.sum(e, axis=1, keepdims=True)


def _pure_fp16(x):
    y = np.asarray(x, dtype=np.float16)
    y = y - np.max(y, axis=1, keepdims=True)
    e = np.exp(y).astype(np.float16)
    s = np.sum(e, axis=1, keepdims=True, dtype=np.float16)
    return (e / s).astype(np.float16)


def grade(sol, fx) -> dict:
    x = np.array(
        [
            [3200, 3199, 3198, 3197, 3196, 3195],
            [-2800, -2799, -2798, -2797, -2796, -2795],
            [1500, 1499.5, 1498.5, 1497.5, 1496.5, 1495.5],
        ],
        dtype=np.float16,
    )

    ref = _oracle(x)
    fp16 = _pure_fp16(x)

    try:
        got = sol.softmax_fp32(x)
        got = np.asarray(got)
        err = float(np.max(np.abs(got.astype(np.float64) - ref)))
    except Exception:
        return {"max_abs_err": float("inf"), "fp16_is_worse": 0.0}

    fp16_err = float(np.max(np.abs(fp16.astype(np.float64) - ref)))
    return {
        "max_abs_err": err,
        "fp16_is_worse": 1.0 if fp16_err > err else 0.0,
    }
