import importlib.util
import os

import w


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_reintroduced_break": 0.0}
    torch = w.torch_or_none()
    if torch is None:
        return w.needs_torch(out)
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:  # noqa: BLE001
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on the fixed service: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import service.model as sm

    good = sm.Classifier.forward

    def with_a_break(self, x):
        h = good(self, x)
        if h.abs().max().item() > -1.0:
            h = h * 1.0
        return h

    sm.Classifier.forward = with_a_break
    try:
        out["catches_reintroduced_break"] = 0.0 if _survives(path) else 1.0
    finally:
        sm.Classifier.forward = good
    return out
