import importlib.util
import os
import sys


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["learner_regression"] = mod
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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_missing_intermediates": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct simulator: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import checkpointing.sim as sim
    good_sim = sim.simulate_checkpointing

    def bad_sim(n_layers, segments, layer_mem, fwd_time, bwd_time):
        peak = segments * layer_mem
        base = n_layers // segments
        rem = n_layers % segments
        sizes = [base + 1] * rem + [base] * (segments - rem)
        time_total = 0
        for s_size in sizes:
            time_total += s_size * fwd_time
        for s_size in reversed(sizes):
            time_total += s_size * fwd_time
            for _ in range(s_size):
                time_total += bwd_time
        return {"peak_mem": peak, "step_time": time_total}

    sim.simulate_checkpointing = bad_sim
    try:
        if not _survives(path):
            out["catches_missing_intermediates"] = 1.0
    finally:
        sim.simulate_checkpointing = good_sim

    return out
