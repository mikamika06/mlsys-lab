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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_bpw": 0.0, "catches_broken_selector": 0.0, "faults_caught": 0.0}
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import quant.analyzer as qa
    import quant.selector as qs

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {e}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_bpw = qa.compute_bpw_and_size
    qa.compute_bpw_and_size = lambda p, b: (b, 0)
    try:
        out["catches_broken_bpw"] = 0.0 if _survives(path) else 1.0
    finally:
        qa.compute_bpw_and_size = good_bpw

    good_select = qs.auto_select_recipe
    qs.auto_select_recipe = lambda ram, tbl: "none"
    try:
        out["catches_broken_selector"] = 0.0 if _survives(path) else 1.0
    finally:
        qs.auto_select_recipe = good_select

    out["faults_caught"] = out["catches_broken_bpw"] + out["catches_broken_selector"]
    return out
