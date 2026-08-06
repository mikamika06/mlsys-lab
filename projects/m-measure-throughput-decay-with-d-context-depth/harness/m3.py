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
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_decay_bug": 0.0,
        "catches_offload_bug": 0.0,
    }
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
        out["_note"] = "No test_* functions found in test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import llamaperf.decay as d_mod
    import llamaperf.offload as o_mod

    good_decay = d_mod.measure_context_decay
    good_offload = o_mod.compare_offload_throughput

    def broken_decay(config, depths):
        res = good_decay(config, depths)
        n = len(depths)
        base = res["throughputs"][0] if res["throughputs"] else 100.0
        res["throughputs"] = [base] * n
        res["decay_ratios"] = [1.0] * n
        return res

    def broken_offload(config, depth, ngl1, ngl2):
        return {
            "throughput_ngl1": 100.0,
            "throughput_ngl2": 100.0,
            "speedup": 1.0,
            "offload_gain_tok_s": 0.0,
        }

    try:
        d_mod.measure_context_decay = broken_decay
        out["catches_decay_bug"] = 0.0 if _survives(path) else 1.0
    finally:
        d_mod.measure_context_decay = good_decay

    try:
        o_mod.compare_offload_throughput = broken_offload
        out["catches_offload_bug"] = 0.0 if _survives(path) else 1.0
    finally:
        o_mod.compare_offload_throughput = good_offload

    return out
