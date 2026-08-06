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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_isolation": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {e}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import prefixhash.salt as s
    good = s.verify_salt_isolation

    def broken(blocks_a, blocks_b, salt_a, salt_b):
        return True

    s.verify_salt_isolation = broken
    import prefixhash
    try:
        prefixhash.verify_salt_isolation = broken
    except AttributeError:
        pass

    try:
        out["catches_broken_isolation"] = 0.0 if _survives(path) else 1.0
    finally:
        s.verify_salt_isolation = good
        try:
            prefixhash.verify_salt_isolation = good
        except AttributeError:
            pass
    return out
