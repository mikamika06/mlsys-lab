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

def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bad_latency": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import moe_sim.metrics as m
    good_latency = m.latency

    def bad_latency(cfg, ngl, n_cpu_experts):
        exp_cpu = min(cfg["top_k"], n_cpu_experts)
        exp_gpu = cfg["top_k"] - exp_cpu
        gpu_layer_time = cfg["time_base_gpu"] + exp_gpu * cfg["time_exp_gpu"] + exp_cpu * cfg["time_exp_cpu"]
        cpu_layer_time = cfg["time_base_cpu"] + cfg["top_k"] * cfg["time_exp_cpu"]
        return ngl * gpu_layer_time + (cfg["layers"] - ngl) * cpu_layer_time

    m.latency = bad_latency
    try:
        survives = False
        try:
            survives = _run(path) is True
        except Exception:
            pass
        if not survives:
            out["catches_bad_latency"] = 1.0
    finally:
        m.latency = good_latency

    return out
