import importlib.util
import os
import sys

def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["learner_regression"] = mod
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
    sys.path.insert(0, workdir)
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_mask": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on good code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    try:
        import speculative.acceptance as acc
        good = acc.compute_accepted_length
    except ImportError:
        out["_note"] = "failed to import speculative.acceptance"
        return out

    def bad(draft_tokens, target_probs, draft_probs, grammar_masks, random_samples):
        # Invariant break: ignores grammar masks entirely
        accepted = 0
        for i, token in enumerate(draft_tokens):
            p = target_probs[i][token]
            q = draft_probs[i][token]
            if p >= q or random_samples[i] < (p / q):
                accepted += 1
            else:
                break
        return accepted

    acc.compute_accepted_length = bad
    try:
        if not _survives(path):
            out["catches_broken_mask"] = 1.0
    finally:
        acc.compute_accepted_length = good

    return out
