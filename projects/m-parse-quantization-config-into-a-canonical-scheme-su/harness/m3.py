import importlib.util
import os
import sys


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_invalid_shapes": 0.0}
    sys.path.insert(0, workdir)
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests failed on good reference: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import quantlib.scheme as scheme_mod
    good_compute = scheme_mod.compute_packed_shape

    def buggy_compute_packed_shape(shape, num_bits, group_size, axis=-1):
        axis = axis % len(shape)
        elems_per_int32 = 32 // num_bits
        out_shape = list(shape)
        out_shape[axis] = shape[axis] // elems_per_int32
        return tuple(out_shape)

    scheme_mod.compute_packed_shape = buggy_compute_packed_shape
    try:
        out["catches_invalid_shapes"] = 0.0 if _survives(path) else 1.0
    finally:
        scheme_mod.compute_packed_shape = good_compute

    return out
