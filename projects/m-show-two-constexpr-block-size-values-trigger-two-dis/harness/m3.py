import importlib.util
import os
import sys


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [
        getattr(mod, n)
        for n in dir(mod)
        if n.startswith("test_") and callable(getattr(mod, n))
    ]
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
    sys.path.insert(0, workdir)
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_cache_bypass": 0.0,
    }

    test_path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(test_path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        res = _run(test_path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = (
            f"Tests failed on valid reference code: {type(e).__name__}: {e}"
        )
        return out

    if res is None:
        out["_note"] = "No test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import triton_cache.cache_demo as demo

    orig_inspect = demo.inspect_cache_keys

    def broken_inspect_cache_keys(kernel_fn, block_size_a, block_size_b):
        return [(128,)]

    demo.inspect_cache_keys = broken_inspect_cache_keys

    try:
        survived = _survives(test_path)
        out["catches_cache_bypass"] = 0.0 if survived else 1.0
    finally:
        demo.inspect_cache_keys = orig_inspect

    return out
