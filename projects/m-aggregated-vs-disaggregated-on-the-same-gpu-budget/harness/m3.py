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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_zero_transfer_bug": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import disagg.simulator as sim
    good_disagg = sim.simulate_disaggregated

    def buggy_disagg(requests, num_prefill_gpus, num_decode_gpus, prefill_rate, decode_rate, kv_transfer_rate, bytes_per_token=1024):
        return good_disagg(requests, num_prefill_gpus, num_decode_gpus, prefill_rate, decode_rate, kv_transfer_rate=1e30, bytes_per_token=0)

    sim.simulate_disaggregated = buggy_disagg
    try:
        out["catches_zero_transfer_bug"] = 0.0 if _survives(path) else 1.0
    finally:
        sim.simulate_disaggregated = good_disagg

    return out
