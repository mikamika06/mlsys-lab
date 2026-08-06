import os
import sys
import importlib.util

def _run(path):
    spec = importlib.util.spec_from_file_location("learner_test", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        raise
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
    if not fns:
        return None
    for fn in fns:
        fn()
    return True

def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        return out

    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"failed on good code: {e}"
        return out

    if first is None:
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import longctx_eval.diagnostics as diag
    good = diag.diagnose_models

    def broken(*args, **kwargs):
        res = good(*args, **kwargs)
        for r in res:
            if r['mode'] == 'dilution':
                r['mode'] = 'rope'
        return res

    diag.diagnose_models = broken
    try:
        survived = False
        try:
            survived = _run(path) is True
        except Exception:
            survived = False
        out["catches_broken"] = 0.0 if survived else 1.0
    finally:
        diag.diagnose_models = good

    return out
