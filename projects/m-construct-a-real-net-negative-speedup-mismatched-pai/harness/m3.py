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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_lookup": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")

    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    sys.path.insert(0, workdir)
    import spec_fail.pathology as p

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

    good = p.prompt_lookup_draft

    def broken_draft(sequence, gamma):
        n = len(sequence)
        if n < 2: return []
        last = sequence[-1]
        best_i = -1
        for i in range(n - 1):
            if sequence[i] == last:
                best_i = i
        if best_i == -1: return []
        return sequence[best_i+1 : best_i+1+gamma]

    p.prompt_lookup_draft = broken_draft
    try:
        if not _survives(path):
            out["catches_broken_lookup"] = 1.0
    finally:
        p.prompt_lookup_draft = good

    return out
