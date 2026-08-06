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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_ignored_queue_delay": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct plan: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import tritondrain.timeout as tmod
    good_timeout = tmod.derive_minimum_drain_timeout

    def faulty_timeout(config):
        durations = []
        for req in config.get("requests", []):
            st = req.get("stage")
            rem = req.get("remaining_ms", 0)
            if st == "completed":
                continue
            durations.append(rem)
        if not durations:
            return 0
        return max(durations)

    tmod.derive_minimum_drain_timeout = faulty_timeout
    import tritondrain
    tritondrain.derive_minimum_drain_timeout = faulty_timeout

    try:
        out["catches_ignored_queue_delay"] = 0.0 if _survives(path) else 1.0
    finally:
        tmod.derive_minimum_drain_timeout = good_timeout
        tritondrain.derive_minimum_drain_timeout = good_timeout

    return out
