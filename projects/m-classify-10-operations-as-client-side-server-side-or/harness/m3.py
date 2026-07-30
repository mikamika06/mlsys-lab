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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_trust_leak": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:  # noqa: BLE001
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct classification: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import opside
    import opside.classify as c
    good_operation = c.classify_operation
    good_all = c.classify_all

    def broken_operation(op):
        return "server" if op["needs_durable_state"] else "client"

    def broken_all(config):
        return [{"name": op["name"], "side": broken_operation(op)} for op in config["operations"]]

    c.classify_operation = broken_operation
    c.classify_all = broken_all
    opside.classify_operation = broken_operation
    opside.classify_all = broken_all
    try:
        out["catches_trust_leak"] = 0.0 if _survives(path) else 1.0
    finally:
        c.classify_operation = good_operation
        c.classify_all = good_all
        opside.classify_operation = good_operation
        opside.classify_all = good_all
    return out
