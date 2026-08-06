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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_ignored_overhead": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import offload.simulator as sim

    good_fn = sim.compute_breakeven_prefix_length

    def broken_fn(config, hw, tier):
        c_comp = (2 * config["num_params"]) / (hw["gpu_tflops"] * 1e12)
        o_comp = hw.get("launch_overhead_s", 0.0)
        bytes_per_token = 2 * config["num_layers"] * config["num_kv_heads"] * config["head_dim"] * config["dtype_bytes"]
        c_trans = bytes_per_token / (tier["bandwidth_gbps"] * 1e9)
        o_trans = 0.0
        denom = c_comp - c_trans
        if abs(denom) < 1e-12:
            return float("inf")
        l_be = (o_trans - o_comp) / denom
        return max(0.0, l_be)

    sim.compute_breakeven_prefix_length = broken_fn
    try:
        out["catches_ignored_overhead"] = 0.0 if _survives(path) else 1.0
    finally:
        sim.compute_breakeven_prefix_length = good_fn

    return out
