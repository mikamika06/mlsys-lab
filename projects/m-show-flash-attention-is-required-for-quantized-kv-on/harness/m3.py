import importlib.util
import os
import sys
import numpy as np


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
    sys.path.insert(0, workdir)
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_materialization_leaks": 0.0}

    test_path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(test_path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        res = _run(test_path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests failed on good code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if res is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import kvquant.attention as attn_mod
    orig_unfused = attn_mod.unfused_attn_q8_0

    def broken_unfused(q, k_qdict, v_qdict, sm_scale=1.0):
        out_arr, _ = orig_unfused(q, k_qdict, v_qdict, sm_scale)
        fake_bytes = 0
        return out_arr, fake_bytes

    attn_mod.unfused_attn_q8_0 = broken_unfused
    try:
        failed = not _survives(test_path)
        out["catches_materialization_leaks"] = 1.0 if failed else 0.0
    finally:
        attn_mod.unfused_attn_q8_0 = orig_unfused

    return out
