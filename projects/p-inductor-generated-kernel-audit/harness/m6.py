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
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_broken_cache": 0.0,
        "catches_broken_fusion": 0.0,
        "faults_caught": 0.0,
    }
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import audit.config as cfg_mod
    import audit.core as core_mod

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {e}"
        return out

    if first is None:
        out["_note"] = "no test functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_cache = cfg_mod.get_default_config

    def broken_cache():
        return {"cache_enabled": False}

    cfg_mod.get_default_config = broken_cache
    try:
        out["catches_broken_cache"] = 0.0 if _survives(path) else 1.0
    finally:
        cfg_mod.get_default_config = good_cache

    good_fusion = core_mod.analyze_fusion_gap

    def broken_fusion(code, mode):
        return {"reason": "none", "fused": True}

    core_mod.analyze_fusion_gap = broken_fusion
    try:
        out["catches_broken_fusion"] = 0.0 if _survives(path) else 1.0
    finally:
        core_mod.analyze_fusion_gap = good_fusion

    out["faults_caught"] = out["catches_broken_cache"] + out["catches_broken_fusion"]
    return out
