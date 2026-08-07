import importlib.util
import os
import sys


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod)
           if n.startswith("test_") and callable(getattr(mod, n))]
    if not fns:
        return None
    for fn in fns:
        fn()
    return True


def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False


def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_invalid_broadcast": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import onnxcalc.broadcast as b
    import onnxcalc.checker as c
    import onnxcalc.value_info as v

    good_broadcast = b.compute_broadcast_shape

    def broken_broadcast(shape_a, shape_b):
        sa = list(shape_a)
        sb = list(shape_b)
        max_rank = max(len(sa), len(sb))
        pa = sa + [1] * (max_rank - len(sa))
        pb = sb + [1] * (max_rank - len(sb))
        out_dims = []
        for da, db in zip(pa, pb):
            if da == 1:
                out_dims.append(db)
            elif db == 1:
                out_dims.append(da)
            else:
                out_dims.append(da)
        return out_dims

    b.compute_broadcast_shape = broken_broadcast
    v.compute_broadcast_shape = broken_broadcast
    if hasattr(c, "compute_broadcast_shape"):
        c.compute_broadcast_shape = broken_broadcast

    try:
        out["catches_invalid_broadcast"] = 0.0 if _survives(path) else 1.0
    finally:
        b.compute_broadcast_shape = good_broadcast
        v.compute_broadcast_shape = good_broadcast
        if hasattr(c, "compute_broadcast_shape"):
            c.compute_broadcast_shape = good_broadcast

    return out
