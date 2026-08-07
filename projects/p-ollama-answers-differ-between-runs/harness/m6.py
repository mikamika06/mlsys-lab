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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_unfixed_seed": 0.0, "catches_missing_params": 0.0, "faults_caught": 0.0}
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import runner.client as rc

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {type(e).__name__}: {e}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    orig_gen = rc.ChatClient.generate
    def flaky_gen(self, prompt, **kwargs):
        import random
        return f"response_{random.randint(1, 100)}"
    rc.ChatClient.generate = flaky_gen
    try:
        out["catches_unfixed_seed"] = 0.0 if _survives(path) else 1.0
    finally:
        rc.ChatClient.generate = orig_gen

    orig_merge = rc.merge_options if hasattr(rc, "merge_options") else None
    def bad_merge(mf, api, req):
        return {}
    rc.merge_options = bad_merge
    try:
        out["catches_missing_params"] = 0.0 if _survives(path) else 1.0
    finally:
        if orig_merge is not None:
            rc.merge_options = orig_merge

    out["faults_caught"] = out["catches_unfixed_seed"] + out["catches_missing_params"]
    return out
