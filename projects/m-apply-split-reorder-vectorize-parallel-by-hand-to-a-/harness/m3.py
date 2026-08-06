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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_missing_vectorization": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct schedule implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import tirsched.schedule as sched_mod
    good_func = sched_mod.apply_split_reorder_vectorize_parallel

    def broken_apply(tir_mod, factors=(16, 16)):
        steps = good_func(tir_mod, factors)
        last_name, last_mod = steps[-1]
        broken_mod = dict(last_mod)
        broken_mod.pop("vectorized", None)
        if "transforms" in broken_mod:
            broken_mod["transforms"] = [t for t in broken_mod["transforms"] if t != "vectorize"]
        steps[-1] = (last_name, broken_mod)
        return steps

    sched_mod.apply_split_reorder_vectorize_parallel = broken_apply
    try:
        out["catches_missing_vectorization"] = 0.0 if _survives(path) else 1.0
    finally:
        sched_mod.apply_split_reorder_vectorize_parallel = good_func

    return out
