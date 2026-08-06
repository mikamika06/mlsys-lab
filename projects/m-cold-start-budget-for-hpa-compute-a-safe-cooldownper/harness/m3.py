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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_optimistic_cooldown": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"The tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import hpabudget.cooldown as cd
    good_cooldown = cd.compute_safe_cooldown_period

    def broken_cooldown(phase_times, warm_compile_cache, cache_speedup_factor, safety_margin_pct):
        raw = good_cooldown(phase_times, warm_compile_cache, cache_speedup_factor, safety_margin_pct)
        return int(raw * 0.5)

    cd.compute_safe_cooldown_period = broken_cooldown
    import hpabudget
    hpabudget.cooldown.compute_safe_cooldown_period = broken_cooldown

    try:
        out["catches_optimistic_cooldown"] = 0.0 if _survives(path) else 1.0
    finally:
        cd.compute_safe_cooldown_period = good_cooldown
        hpabudget.cooldown.compute_safe_cooldown_period = good_cooldown

    return out
