import sys

import numpy as np

from mlsys import scorers

FAIL = {"byte_exact_fraction": 0.0, "zero_copy_fraction": 0.0}

_FORBIDDEN = {
    "broadcast_arrays", "_broadcast_arrays",
    "broadcast_to", "_broadcast_to",
    "broadcast_shapes", "_broadcast_shape",
    "tile", "repeat", "resize",
}


def _uses_forbidden(fn):
    hit = {"bad": False}

    def tracer(frame, event, arg):
        if event == "call":
            mod = frame.f_globals.get("__name__", "")
            if mod.startswith("numpy") and frame.f_code.co_name in _FORBIDDEN:
                hit["bad"] = True
        return tracer

    prev = sys.gettrace()
    sys.settrace(tracer)
    try:
        result = fn()
    finally:
        sys.settrace(prev)
    return hit["bad"], result


def _cases():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(4, 6))
    return [
        (np.arange(3.0), rng.normal(size=(4, 3))),
        (rng.normal(size=(5, 1)), rng.normal(size=(1, 6))),
        (np.array(2.5), rng.normal(size=(3, 4))),
        (rng.normal(size=(2, 1, 4)), rng.normal(size=(3, 1))),
        (rng.normal(size=(1, 5)), np.arange(5.0)),
        (rng.normal(size=(4, 1, 3)), rng.normal(size=(2, 1))),
        (rng.integers(0, 100, size=(1, 3)).astype(np.int32),
         rng.integers(0, 100, size=(4, 1)).astype(np.int32)),
        (base[:, ::2], rng.normal(size=(4, 3))),   # non-contiguous source
    ]


def grade(sol, fx) -> dict:
    cases = _cases()
    n_pairs = 0
    byte_hits = 0.0
    view_hits = 0

    for a, b in cases:
        try:
            bad, result = _uses_forbidden(lambda: sol.broadcast_pair(a, b))
            if bad:
                return dict(FAIL)
            got_a, got_b = result
        except Exception:
            return dict(FAIL)

        ref_a, ref_b = np.broadcast_arrays(a, b)   # real NumPy oracle

        if not (isinstance(got_a, np.ndarray) and isinstance(got_b, np.ndarray)):
            n_pairs += 2
            continue
        if got_a.shape != ref_a.shape or got_b.shape != ref_b.shape:
            n_pairs += 2
            continue

        byte_hits += scorers.byte_exact_fraction(
            np.ascontiguousarray(got_a), np.ascontiguousarray(ref_a))
        byte_hits += scorers.byte_exact_fraction(
            np.ascontiguousarray(got_b), np.ascontiguousarray(ref_b))
        n_pairs += 2

        if np.shares_memory(got_a, a) and got_a.strides == ref_a.strides:
            view_hits += 1
        if np.shares_memory(got_b, b) and got_b.strides == ref_b.strides:
            view_hits += 1

    return {
        "byte_exact_fraction": float(byte_hits / n_pairs),
        "zero_copy_fraction": float(view_hits / n_pairs),
    }
