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


def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_warm_leakage": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        if _run(path) is not True:
            out["_note"] = "No valid test functions found"
            return out
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests failed on reference solution: {e}"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import coldcache.protocol as p
    orig_reset = p.ColdCacheProtocol.reset_gpu_allocator

    def leaky_reset(self):
        return self.generation

    p.ColdCacheProtocol.reset_gpu_allocator = leaky_reset

    try:
        try:
            passed_faulty = _run(path)
        except Exception:
            passed_faulty = False

        if not passed_faulty:
            out["catches_warm_leakage"] = 1.0
        else:
            out["_note"] = "Learner's tests failed to catch an un-incremented allocator generation fault"
    finally:
        p.ColdCacheProtocol.reset_gpu_allocator = orig_reset

    return out
