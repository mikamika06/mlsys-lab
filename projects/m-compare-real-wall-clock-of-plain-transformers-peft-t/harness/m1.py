import importlib.util
import os
import ref

def check(workdir):
    out = {"profile_runs_matched": 0.0, "latency_ratio": 0.0}

    path = os.path.join(workdir, "bench", "profile.py")
    if not os.path.isfile(path):
        out["_note"] = "bench/profile.py not found"
        return out

    spec = importlib.util.spec_from_file_location("learner_profile", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    pt_step, mlx_step = ref.create_mock_step_fns(pt_delay=0.002, mlx_delay=0.001)
    try:
        res = mod.profile_runtimes(pt_step, mlx_step, steps=5, warmup_steps=1)
    except Exception as e:
        out["_note"] = f"profile_runtimes failed: {type(e).__name__}: {str(e)}"
        return out

    required_keys = {"pt_total_sec", "mlx_total_sec", "latency_ratio", "pt_avg_step_sec", "mlx_avg_step_sec"}
    if not required_keys.issubset(res.keys()):
        out["_note"] = f"Missing required metrics in result dictionary. Got: {list(res.keys())}"
        return out

    out["profile_runs_matched"] = 1.0
    out["latency_ratio"] = float(res["latency_ratio"])
    return out
