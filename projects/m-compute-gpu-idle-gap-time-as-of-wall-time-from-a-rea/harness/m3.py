import importlib.util
import os
import ref


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bad_sync": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        res = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests failed on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if res is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import nsysprof.syncs as s
    good_fn = s.count_sync_points

    def bad_fn(trace):
        return 0

    s.count_sync_points = bad_fn
    try:
        failed = False
        try:
            _run(path)
        except Exception:
            failed = True
        out["catches_bad_sync"] = 1.0 if failed else 0.0
    finally:
        s.count_sync_points = good_fn

    return out
