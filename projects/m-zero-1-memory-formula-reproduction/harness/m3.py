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
        "catches_unsharded_states": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = (
            f"tests fail on reference code: {type(e).__name__}: {str(e)[:120]}"
        )
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import zero1.memory as m

    good_calc = m.calculate_zero1_memory

    def broken_unsharded_calc(num_params, world_size, precision_bytes=2):
        res = good_calc(num_params, world_size, precision_bytes)
        res["opt_state_per_rank_bytes"] = res["opt_state_per_rank_bytes"] * world_size
        res["zero1_bytes"] = res["baseline_bytes"]
        return res

    m.calculate_zero1_memory = broken_unsharded_calc
    try:
        out["catches_unsharded_states"] = 0.0 if _survives(path) else 1.0
    finally:
        m.calculate_zero1_memory = good_calc

    return out
