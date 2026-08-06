import importlib.util
import os


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod) if (n.startswith("test_") or n == "test_regression") and callable(getattr(mod, n))]
    for fn in fns:
        fn()
    return True


def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False


def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_decision": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        res = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests failed on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import minja_tools.decision as d
    orig = d.ToolDecisionEngine.needs_jinja

    def broken(self, template_str, tools_present):
        return False

    d.ToolDecisionEngine.needs_jinja = broken
    try:
        survived = _survives(path)
        out["catches_broken_decision"] = 0.0 if survived else 1.0
    finally:
        d.ToolDecisionEngine.needs_jinja = orig
    return out
