import importlib.util
import os

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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bad_stride": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests failed on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import quantpack.layout as l
    good_strides = l.get_memory_strides

    def bad_strides(shape, dtype_size=4):
        return tuple(0 for _ in shape)

    l.get_memory_strides = bad_strides
    import quantpack
    if hasattr(quantpack, "layout"):
        quantpack.layout.get_memory_strides = bad_strides
    try:
        out["catches_bad_stride"] = 0.0 if _survives(path) else 1.0
    finally:
        l.get_memory_strides = good_strides
        if hasattr(quantpack, "layout"):
            quantpack.layout.get_memory_strides = good_strides
    return out
