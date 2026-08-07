import ref
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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_mask": 0.0, "catches_broken_budget": 0.0}
    if not os.path.isfile(path):
        return out

    try:
        first = _run(path)
    except Exception:
        out["has_tests"] = 1.0
        return out

    if first is None:
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import app.decoder as dec
    good_mask = dec.apply_grammar_mask
    dec.apply_grammar_mask = lambda logits, allowed: logits
    try:
        out["catches_broken_mask"] = 0.0 if _survives(path) else 1.0
    finally:
        dec.apply_grammar_mask = good_mask

    import app.components as comp
    good_closing = comp.MockFSM.get_closing_tokens
    comp.MockFSM.get_closing_tokens = lambda self: []
    try:
        out["catches_broken_budget"] = 0.0 if _survives(path) else 1.0
    finally:
        comp.MockFSM.get_closing_tokens = good_closing

    return out
