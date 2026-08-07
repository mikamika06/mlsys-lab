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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_tactic_leak": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct check: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import trtcache.parser as p
    good = p.is_reusable

    def leaky_reusable(cache_header, builder_config):
        if cache_header["version"] != builder_config["version"]:
            return False
        if cache_header["sm_major"] != builder_config["sm_major"]:
            return False
        if cache_header["sm_minor"] != builder_config["sm_minor"]:
            return False
        if cache_header["opt_level"] != builder_config["opt_level"]:
            return False
        return True

    p.is_reusable = leaky_reusable
    try:
        out["catches_tactic_leak"] = 0.0 if _survives(path) else 1.0
    finally:
        p.is_reusable = good

    return out
