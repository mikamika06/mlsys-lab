import sys

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import importlib
    engine_mod = importlib.import_module("runner.engine")
    profiler_mod = importlib.import_module("runner.profiler")

    out = {"slots_bottleneck_correct": 0.0, "batch_bottleneck_correct": 0.0}

    try:
        cfg_slots = engine_mod.EngineConfig(gpu_memory_mb=3072, bytes_per_slot_mb=1024, max_batch_size=8)
        b_slots = profiler_mod.identify_bottleneck(cfg_slots, active_users=4)
        if b_slots == "SLOTS":
            out["slots_bottleneck_correct"] = 1.0
    except Exception:
        pass

    try:
        cfg_batch = engine_mod.EngineConfig(gpu_memory_mb=8192, bytes_per_slot_mb=512, max_batch_size=2)
        b_batch = profiler_mod.identify_bottleneck(cfg_batch, active_users=4)
        if b_batch == "BATCH":
            out["batch_bottleneck_correct"] = 1.0
    except Exception:
        pass

    return out
