import importlib.util
import math
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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_uncapped_allocations": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:  # noqa: BLE001
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import kvtrace.simulator as sim
    import kvtrace.analysis as ana

    good_trace = sim.trace_block_timeline
    good_waste = ana.compute_paged_waste

    def uncapped_trace(events, block_size):
        timeline = good_trace(events, block_size)
        return [(t, blocks * 10) for t, blocks in timeline]

    def uncapped_waste(length_histogram, block_size):
        return -100

    sim.trace_block_timeline = uncapped_trace
    ana.compute_paged_waste = uncapped_waste

    import kvtrace

    kvtrace.simulator.trace_block_timeline = uncapped_trace
    kvtrace.analysis.compute_paged_waste = uncapped_waste

    try:
        out["catches_uncapped_allocations"] = 0.0 if _survives(path) else 1.0
    finally:
        sim.trace_block_timeline = good_trace
        ana.compute_paged_waste = good_waste
        kvtrace.simulator.trace_block_timeline = good_trace
        kvtrace.analysis.compute_paged_waste = good_waste

    return out
