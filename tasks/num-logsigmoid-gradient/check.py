import numpy as np


def _ref_value(x):
    x = np.asarray(x, dtype=np.float64)
    return -np.logaddexp(0.0, -x)


def _ref_grad(x):
    x = np.asarray(x, dtype=np.float64)
    h = 1e-4
    return (_ref_value(x + h) - _ref_value(x - h)) / (2.0 * h)


def grade(sol, fx) -> dict:
    x = np.array(
        [
            -1000.0,
            -50.0,
            -20.0,
            -5.5,
            -1.0,
            -0.25,
            0.0,
            0.25,
            1.0,
            5.5,
            20.0,
            50.0,
            1000.0,
        ],
        dtype=np.float64,
    )

    try:
        value, grad = sol.logsigmoid_with_grad(x)
        value = np.asarray(value, dtype=np.float64)
        grad = np.asarray(grad, dtype=np.float64)
    except Exception:
        return {
            "value_max_abs_err": float("inf"),
            "grad_max_abs_err": float("inf"),
        }

    return {
        "value_max_abs_err": float(np.max(np.abs(value - _ref_value(x)))),
        "grad_max_abs_err": float(np.max(np.abs(grad - _ref_grad(x)))),
    }
