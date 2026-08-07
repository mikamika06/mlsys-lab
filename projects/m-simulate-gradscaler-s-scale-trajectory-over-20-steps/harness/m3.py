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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_unreset_successes": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:  # noqa: BLE001
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct simulator: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import gradscaler.sim as gs
    good_sim = gs.simulate_trajectory

    def faulty_sim(overflows, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000):
        scales = []
        current_scale = float(init_scale)
        successes = 0
        for is_overflow in overflows:
            scales.append(current_scale)
            if is_overflow:
                current_scale *= backoff_factor
                # INJECTED FAULT: forgot to reset successes to 0
            else:
                successes += 1
                if successes == growth_interval:
                    current_scale *= growth_factor
                    successes = 0
        return scales

    gs.simulate_trajectory = faulty_sim

    import gradscaler
    gradscaler.simulate_trajectory = faulty_sim

    try:
        out["catches_unreset_successes"] = 0.0 if _survives(path) else 1.0
    finally:
        gs.simulate_trajectory = good_sim
        gradscaler.simulate_trajectory = good_sim

    return out
