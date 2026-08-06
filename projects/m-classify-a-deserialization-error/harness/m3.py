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


def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_penalty_bypass": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    sys.path.insert(0, workdir)
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on reference code: {type(e).__name__}: {str(e)[:120]}"
        return out
    finally:
        if workdir in sys.path:
            sys.path.remove(workdir)

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import trtplan.classifier as c
    orig_classify = c.classify_engine

    def faulty_classify(header, runtime_env):
        res = orig_classify(header, runtime_env)
        if res["status"] == "OK" and res["penalty"] is not None and res["penalty"] > 1.0:
            res["penalty"] = 1.0
        return res

    c.classify_engine = faulty_classify
    import trtplan
    trtplan.classify_engine = faulty_classify

    try:
        survived = False
        try:
            survived = (_run(path) is True)
        except Exception:
            survived = False
        out["catches_penalty_bypass"] = 0.0 if survived else 1.0
    finally:
        c.classify_engine = orig_classify
        trtplan.classify_engine = orig_classify

    return out
