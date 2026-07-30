import importlib.util
import os


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_quantized_norms": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct plan: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import diskplan.schemes as s
    good = s.tensor_bytes

    def broken(tensor, scheme):
        bits = scheme["bits"]
        if bits >= 16:
            return tensor["count"] * 2
        weight_bytes = -(-(tensor["count"] * bits) // 8)
        group_size = scheme["group_size"]
        groups = 1 if not group_size else -(-tensor["count"] // group_size)
        return weight_bytes + groups * s.SCALE_BYTES

    s.tensor_bytes = broken
    import diskplan
    diskplan.tensor_bytes = broken
    try:
        out["catches_quantized_norms"] = 0.0 if _survives(path) else 1.0
    finally:
        s.tensor_bytes = good
        diskplan.tensor_bytes = good
    return out
