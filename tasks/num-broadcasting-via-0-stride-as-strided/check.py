import sys

import numpy as np

from mlsys import scorers

FAIL = {
    "byte_exact_fraction": 0.0,
    "zero_copy_view_fraction": 0.0,
    "raises_on_bad_shape_fraction": 0.0,
    "cases": 0.0,
}

_FORBIDDEN = {"broadcast_to", "_broadcast_to", "broadcast_arrays",
              "_broadcast_arrays", "tile", "repeat", "resize"}


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
        fn()
    finally:
        sys.settrace(prev)
    return hit["bad"]


def _cases():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(4, 6))
    strided_src = base[:, ::2]                  # non-contiguous (4, 3)
    out = [
        (np.arange(3.0), (4, 3)),
        (rng.normal(size=(1, 5)), (3, 5)),
        (rng.normal(size=(3, 1)), (3, 7)),
        (np.array(2.5), (2, 3)),
        (strided_src[:, None, :], (4, 5, 3)),
        (rng.normal(size=(2, 1, 4)), (2, 3, 4)),
        (rng.integers(0, 100, size=(1, 3)).astype(np.int32), (1, 3)),
        (rng.normal(size=(5,)), (5,)),
    ]
    return out


_BAD = [
    (np.zeros((3, 2)), (3, 5)),
    (np.zeros((4,)), (3,)),
    (np.zeros((2, 3)), (3,)),
]


def grade(sol, fx) -> dict:
    cases = _cases()
    n_bytes_ok = 0
    n_view_ok = 0

    for a, shape in cases:
        try:
            if _uses_forbidden(lambda: sol.broadcast_to_strided(a, shape)):
                return dict(FAIL)
            got = sol.broadcast_to_strided(a, shape)
        except Exception:
            return dict(FAIL)

        ref = np.broadcast_to(a, shape)  # real NumPy oracle

        if not isinstance(got, np.ndarray):
            continue
        if got.shape != ref.shape or got.dtype != ref.dtype:
            continue
        if scorers.byte_exact_fraction(np.ascontiguousarray(got),
                                       np.ascontiguousarray(ref)) == 1.0:
            n_bytes_ok += 1
        if np.shares_memory(got, a) and got.strides == ref.strides:
            n_view_ok += 1

    n_raise_ok = 0
    for a, shape in _BAD:
        try:
            sol.broadcast_to_strided(a, shape)
        except ValueError:
            n_raise_ok += 1
        except Exception:
            pass

    n = len(cases)
    return {
        "byte_exact_fraction": float(n_bytes_ok / n),
        "zero_copy_view_fraction": float(n_view_ok / n),
        "raises_on_bad_shape_fraction": float(n_raise_ok / len(_BAD)),
        "cases": float(n),
    }
