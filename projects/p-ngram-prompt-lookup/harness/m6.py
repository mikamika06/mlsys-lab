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
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {"has_tests": 0.0, "passes_on_good": 0.0,
           "catches_broken_index": 0.0, "catches_broken_policy": 0.0,
           "faults_caught": 0.0}
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import ngram.index as idx_mod
    import ngram.policy as pol_mod

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

    good_lookup = idx_mod.PromptNgramIndex.lookup
    def broken_lookup(self, tokens):
        return []
    idx_mod.PromptNgramIndex.lookup = broken_lookup
    try:
        out["catches_broken_index"] = 0.0 if _survives(path) else 1.0
    finally:
        idx_mod.PromptNgramIndex.lookup = good_lookup

    good_select = pol_mod.select_candidates
    def broken_select(index, tokens, k=4):
        return []
    pol_mod.select_candidates = broken_select
    try:
        out["catches_broken_policy"] = 0.0 if _survives(path) else 1.0
    finally:
        pol_mod.select_candidates = good_select

    out["faults_caught"] = out["catches_broken_index"] + out["catches_broken_policy"]
    return out
