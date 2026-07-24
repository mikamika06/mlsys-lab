import sys

import numpy as np

from mlsys import scorers


def _ref_layernorm(x, gamma, beta, eps):
    """NumPy oracle: fused LayerNorm (normalize + affine) over the last axis.

    Uses the biased/population variance (divide by D), matching PyTorch.
    """
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    mu = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_hat = (x - mu) / np.sqrt(var + eps)
    return gamma * x_hat + beta


def _count_student_line_events(fn, *args, **kwargs) -> int:
    """Count line events executed IN THE STUDENT'S OWN FILE during ``fn(...)``.

    NumPy's C work (and its Python wrappers, which live in other files) is
    ignored, so this isolates the student's interpreted lines: a vectorised
    solution runs a handful, a Python loop over rows/features runs far more.
    """
    filename = fn.__code__.co_filename
    count = 0

    def tracer(frame, event, arg):
        nonlocal count
        if event == "line" and frame.f_code.co_filename == filename:
            count += 1
        return tracer

    prev = sys.gettrace()
    sys.settrace(tracer)
    try:
        fn(*args, **kwargs)
    finally:
        sys.settrace(prev)
    return count


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    N, D = 64, 48
    eps = 1e-5
    x = rng.standard_normal((N, D))
    gamma = rng.standard_normal(D)
    beta = rng.standard_normal(D)

    # Reference computed from the oracle, never hardcoded.
    ref = _ref_layernorm(x, gamma, beta, eps)

    # --- accuracy: max absolute error vs the NumPy reference ---
    try:
        y = sol.layernorm(x, gamma, beta, eps)
        max_err = scorers.max_abs_err(ref, y)
    except Exception:
        max_err = 1e9

    # --- vectorization: Python line events in the student's file ---
    try:
        op_count = float(_count_student_line_events(sol.layernorm, x, gamma, beta, eps))
    except Exception:
        op_count = 1e9

    return {"max_abs_err": max_err, "op_count": op_count}
