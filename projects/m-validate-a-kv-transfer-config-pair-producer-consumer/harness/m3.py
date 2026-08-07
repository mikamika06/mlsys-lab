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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_ignored_transport": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Learner tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found in test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import kvtransfer.config as cfg_mod
    original_validate = cfg_mod.validate_pair

    def broken_validate_pair(producer_cfg, consumer_cfg):
        res = original_validate(producer_cfg, consumer_cfg)
        res["errors"] = [e for e in res["errors"] if e != "mismatched transport type"]
        res["valid"] = len(res["errors"]) == 0
        return res

    cfg_mod.validate_pair = broken_validate_pair
    try:
        out["catches_ignored_transport"] = 0.0 if _survives(path) else 1.0
    finally:
        cfg_mod.validate_pair = original_validate

    return out
