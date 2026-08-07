import sys

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import importlib
    engine_mod = importlib.import_module("runner.engine")
    profiler_mod = importlib.import_module("runner.profiler")

    out = {"curve_ok": 0.0, "knee_correct": 0.0}

    try:
        cfg = engine_mod.EngineConfig()
        slot_counts = [1, 2, 3, 4, 5, 6]
        curve = profiler_mod.build_slot_scaling_curve(cfg, slot_counts)
    except Exception:
        return out

    if isinstance(curve, dict) and all(s in curve for s in slot_counts):
        out["curve_ok"] = 1.0

    try:
        knee = profiler_mod.find_knee(curve)
        if knee == 3:
            out["knee_correct"] = 1.0
    except Exception:
        pass

    return out
