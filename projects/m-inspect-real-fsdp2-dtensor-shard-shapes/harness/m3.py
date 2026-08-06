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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_reshard_bug": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
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

    import fsdpshards.memory as m

    good = m.compute_reshard_memory_profile

    def broken_reshard(layer_param_shapes, mesh_size, dtype_bytes=4, reshard_after_forward=True):
        res = good(layer_param_shapes, mesh_size, dtype_bytes, reshard_after_forward)
        res["saved_bytes_after_forward"] = 0
        res["persistent_param_bytes_after_forward"] = res["peak_param_bytes_during_forward"]
        return res

    m.compute_reshard_memory_profile = broken_reshard
    import fsdpshards

    fsdpshards.compute_reshard_memory_profile = broken_reshard
    try:
        out["catches_reshard_bug"] = 0.0 if _survives(path) else 1.0
    finally:
        m.compute_reshard_memory_profile = good
        fsdpshards.compute_reshard_memory_profile = good

    return out
