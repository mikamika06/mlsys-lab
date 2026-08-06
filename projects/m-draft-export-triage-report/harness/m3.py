import importlib.util
import os


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [
        getattr(mod, n)
        for n in dir(mod)
        if n.startswith("test_") and callable(getattr(mod, n))
    ]
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
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_unclassified_fallbacks": 0.0
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import triage.report as r
    good_gen = r.generate_triage_report

    def broken_gen(triage_results):
        report = good_gen(triage_results)
        report["issue_counts"]["UNKNOWN"] = 0
        return report

    r.generate_triage_report = broken_gen
    import triage
    triage.report.generate_triage_report = broken_gen

    try:
        out["catches_unclassified_fallbacks"] = 0.0 if _survives(path) else 1.0
    finally:
        r.generate_triage_report = good_gen
        triage.report.generate_triage_report = good_gen

    return out
