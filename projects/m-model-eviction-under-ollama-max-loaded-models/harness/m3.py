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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bad_policy": 0.0}
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

    import ollama_evict.policy as p
    good_policy = p.select_evict

    def bad_policy(loaded_models, access_times, max_loaded):
        if len(loaded_models) <= max_loaded:
            return None
        sorted_models = sorted(loaded_models, key=lambda m: access_times.get(m, 0), reverse=True)
        return sorted_models[0]

    p.select_evict = bad_policy
    import ollama_evict
    if hasattr(ollama_evict, "select_evict"):
        ollama_evict.select_evict = bad_policy

    try:
        out["catches_bad_policy"] = 0.0 if _survives(path) else 1.0
    finally:
        p.select_evict = good_policy
        if hasattr(ollama_evict, "select_evict"):
            ollama_evict.select_evict = good_policy
    return out
