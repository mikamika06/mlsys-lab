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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_unaligned_len": 0.0}
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

    import kvmem.solver as s
    from kvmem.config import get_bytes_per_token
    good_len = s.solve_max_model_len

    def unaligned_solve(config, dtype, total_vram, weights_size, util, block_size):
        avail = int(total_vram * util) - weights_size
        if avail <= 0: return 0
        return avail // get_bytes_per_token(config, dtype)

    s.solve_max_model_len = unaligned_solve
    try:
        if not _survives(path):
            out["catches_unaligned_len"] = 1.0
        else:
            out["_note"] = "test did not fail when max_model_len ignores block boundaries"
    finally:
        s.solve_max_model_len = good_len

    return out
