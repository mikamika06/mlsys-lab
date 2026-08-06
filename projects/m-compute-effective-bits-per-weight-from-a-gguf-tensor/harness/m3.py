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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_invalid_choice": 0.0}
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

    import gguf_utils.policy as p
    good_choose = p.choose_model

    def broken_choose(memory_cap_gb, model_options):
        valid = []
        for opt in model_options:
            if opt["memory_gb"] <= memory_cap_gb:
                valid.append(opt)
        if not valid:
            return None
        return min(valid, key=lambda x: x["score"])["name"]

    p.choose_model = broken_choose
    import gguf_utils
    if hasattr(gguf_utils, "policy"):
        gguf_utils.policy.choose_model = broken_choose

    try:
        out["catches_invalid_choice"] = 0.0 if _survives(path) else 1.0
    finally:
        p.choose_model = good_choose
        if hasattr(gguf_utils, "policy"):
            gguf_utils.policy.choose_model = good_choose
    return out
