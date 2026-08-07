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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_invalid_budget": 0.0, "catches_broken_scaling": 0.0, "faults_caught": 0.0}
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import lora_sweep.config as cfg_mod
    import lora_sweep.optimizer as opt_mod

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail: {e}"
        return out
    if first is None:
        out["_note"] = "no test functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_budget = cfg_mod.get_default_config

    def bad_budget():
        c = good_budget()
        c["budget_hours"] = -1.0
        return c

    cfg_mod.get_default_config = bad_budget
    try:
        out["catches_invalid_budget"] = 0.0 if _survives(path) else 1.0
    finally:
        cfg_mod.get_default_config = good_budget

    good_verify = opt_mod.verify_second_domain

    def bad_verify(config):
        return {"verified": False}

    opt_mod.verify_second_domain = bad_verify
    try:
        out["catches_broken_scaling"] = 0.0 if _survives(path) else 1.0
    finally:
        opt_mod.verify_second_domain = good_verify

    out["faults_caught"] = out["catches_invalid_budget"] + out["catches_broken_scaling"]
    return out
