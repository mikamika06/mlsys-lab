import sys

import numpy as np

from mlsys import scorers

# Fixed shapes (num_heads, seq_len, dim); dim divisible by num_heads.
_CASES = [
    (4, 6, 32),
    (2, 5, 16),
    (8, 3, 64),
]


def _ref_split(x, num_heads):
    """Oracle: (T, D) -> (num_heads, T, D // num_heads), head axis first."""
    seq_len, dim = x.shape
    head_dim = dim // num_heads
    # reshape+transpose is the oracle's business; the SOLUTION is what is banned
    # from using it (enforced by the op_count / line-event gate below).
    return np.ascontiguousarray(x.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2))


def _ref_merge(heads):
    """Oracle: (num_heads, T, head_dim) -> (T, num_heads * head_dim)."""
    num_heads, seq_len, head_dim = heads.shape
    return np.ascontiguousarray(heads.transpose(1, 0, 2).reshape(seq_len, num_heads * head_dim))


def _run_traced(fn, *args):
    """Run fn(*args); return (result, python_line_event_count).

    Counts EVERY python line event during the call, across all frames. A pure
    numpy reshape/transpose runs in C and emits ~0 line events; an explicit
    per-element index loop emits ~one event per element. So this is a
    deterministic, hardware-independent discriminator between the two.
    """
    count = 0

    def tracer(frame, event, arg):
        nonlocal count
        if event == "line":
            count += 1
        return tracer

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        result = fn(*args)
    finally:
        sys.settrace(old)
    return result, count


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    max_err = 0.0
    total_ops = 0

    try:
        for num_heads, seq_len, dim in _CASES:
            x = rng.standard_normal((seq_len, dim))

            ref_split = _ref_split(x, num_heads)
            got_split, ops = _run_traced(sol.split_heads, x, num_heads)
            total_ops += ops
            got_split = np.asarray(got_split, dtype=np.float64)
            if got_split.shape != ref_split.shape:
                return {"max_abs_err": float("inf"), "op_count": float(total_ops)}
            max_err = max(max_err, scorers.max_abs_err(ref_split, got_split))

            # feed the oracle's correct split so merge is tested independently
            ref_merge = _ref_merge(ref_split)
            got_merge, ops = _run_traced(sol.merge_heads, ref_split)
            total_ops += ops
            got_merge = np.asarray(got_merge, dtype=np.float64)
            if got_merge.shape != ref_merge.shape:
                return {"max_abs_err": float("inf"), "op_count": float(total_ops)}
            max_err = max(max_err, scorers.max_abs_err(ref_merge, got_merge))
    except Exception:
        return {"max_abs_err": float("inf"), "op_count": 0.0}

    return {"max_abs_err": float(max_err), "op_count": float(total_ops)}
