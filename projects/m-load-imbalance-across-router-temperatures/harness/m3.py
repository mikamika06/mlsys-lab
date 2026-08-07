import importlib.util
import os
import sys


def _run(path):
    spec = importlib.util.spec_from_file_location("test_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
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
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_asymmetric_shapes": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")

    if not os.path.isfile(path):
        out["_note"] = "test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import moe_routing.comm as comm
    good = comm.all_to_all_shapes

    def broken_shapes(assignments, num_experts, num_devices):
        s, r = good(assignments, num_experts, num_devices)
        if num_devices > 1:
            r[0, 1] += 1
        return s, r

    comm.all_to_all_shapes = broken_shapes
    try:
        if not _survives(path):
            out["catches_asymmetric_shapes"] = 1.0
        else:
            out["_note"] = "tests passed even when recv_counts was intentionally corrupted to break symmetry"
    finally:
        comm.all_to_all_shapes = good

    return out
