import sys
import numpy as np


def _oracle_attention(Q, K, V):
    scale = 1.0 / np.sqrt(Q.shape[1])
    scores = (Q @ K.T) * scale
    scores = scores - np.max(scores, axis=1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return weights @ V


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(123)
    Q = rng.normal(size=(8, 4)).astype(np.float64)
    K = rng.normal(size=(8, 4)).astype(np.float64)
    V = rng.normal(size=(8, 3)).astype(np.float64)

    ref = _oracle_attention(Q, K, V)

    allocated_full = False
    old_empty = np.empty
    old_zeros = np.zeros
    old_ones = np.ones
    old_full = np.full

    def watch_shape(shape):
        nonlocal allocated_full
        try:
            if tuple(shape) == (Q.shape[0], Q.shape[0]):
                allocated_full = True
        except TypeError:
            pass

    def empty_hook(*args, **kwargs):
        if args:
            watch_shape(args[0])
        return old_empty(*args, **kwargs)

    def zeros_hook(*args, **kwargs):
        if args:
            watch_shape(args[0])
        return old_zeros(*args, **kwargs)

    def ones_hook(*args, **kwargs):
        if args:
            watch_shape(args[0])
        return old_ones(*args, **kwargs)

    def full_hook(*args, **kwargs):
        if args:
            watch_shape(args[0])
        return old_full(*args, **kwargs)

    line_count = {"n": 0}

    def tracer(frame, event, arg):
        if frame.f_code is getattr(sol.flash_attention_forward, "__code__", None):
            if event == "line":
                line_count["n"] += 1
        return tracer

    np.empty = empty_hook
    np.zeros = zeros_hook
    np.ones = ones_hook
    np.full = full_hook
    old_trace = sys.gettrace()
    sys.settrace(tracer)
    try:
        try:
            got = sol.flash_attention_forward(Q, K, V, 3, 2)
        except Exception:
            return {
                "max_abs_err": float("inf"),
                "tiled_execution": 0.0,
                "no_full_score_buffer": 0.0,
            }
    finally:
        sys.settrace(old_trace)
        np.empty = old_empty
        np.zeros = old_zeros
        np.ones = old_ones
        np.full = old_full

    got = np.asarray(got, dtype=np.float64)
    err = float(np.max(np.abs(got - ref)))
    return {
        "max_abs_err": err,
        "tiled_execution": 1.0 if line_count["n"] >= 20 else 0.0,
        "no_full_score_buffer": 0.0 if allocated_full else 1.0,
    }
