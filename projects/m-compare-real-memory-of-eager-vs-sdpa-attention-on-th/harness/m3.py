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
    sys.path.insert(0, workdir)
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_quadratic_sdpa_bug": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")

    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import memacc.accounting as acc

    good_sdpa = acc.layer_sdpa_memory

    def buggy_sdpa_memory(layer_cfg):
        res = good_sdpa(layer_cfg)
        b = layer_cfg["batch_size"]
        s = layer_cfg["seq_len"]
        h = layer_cfg["num_heads"]
        elem = acc.dtype_bytes(layer_cfg.get("dtype", "float16"))
        res["retained_bytes"] += b * h * s * s * elem
        return res

    acc.layer_sdpa_memory = buggy_sdpa_memory
    import memacc.compare
    memacc.compare.layer_sdpa_memory = buggy_sdpa_memory

    try:
        survived = _survives(path)
        out["catches_quadratic_sdpa_bug"] = 0.0 if survived else 1.0
    finally:
        acc.layer_sdpa_memory = good_sdpa
        memacc.compare.layer_sdpa_memory = good_sdpa

    return out
