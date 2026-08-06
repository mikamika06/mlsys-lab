import importlib.util
import os
import ref

def check(workdir):
    out = {"memory_runs_matched": 0.0, "memory_ratio": 0.0}

    path = os.path.join(workdir, "bench", "memory.py")
    if not os.path.isfile(path):
        out["_note"] = "bench/memory.py not found"
        return out

    spec = importlib.util.spec_from_file_location("learner_memory", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    pt_trainer = ref.MockTrainer(peak_bytes=4 * 1024 * 1024 * 1024)
    mlx_trainer = ref.MockTrainer(peak_bytes=2 * 1024 * 1024 * 1024)

    try:
        res = mod.profile_peak_memory(pt_trainer, mlx_trainer)
    except Exception as e:
        out["_note"] = f"profile_peak_memory failed: {type(e).__name__}: {str(e)}"
        return out

    required_keys = {"pt_peak_bytes", "mlx_peak_bytes", "memory_ratio", "memory_saved_bytes"}
    if not required_keys.issubset(res.keys()):
        out["_note"] = f"Missing required keys in memory result. Got: {list(res.keys())}"
        return out

    out["memory_runs_matched"] = 1.0
    out["memory_ratio"] = float(res["memory_ratio"])
    return out
