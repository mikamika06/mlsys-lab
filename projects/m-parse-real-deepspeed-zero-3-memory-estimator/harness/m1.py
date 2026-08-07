import ref


def check(workdir):
    try:
        from zero_estimator.memory import calculate_sharded_elements, estimate_zero3_memory
    except ImportError:
        return {"_note": "failed to import functions"}

    out = {"sharded_exact": 0.0, "memory_exact": 0.0}
    sharded_ok = 0
    mem_ok = 0

    for i, cfg in enumerate(ref.CONFIGS):
        lp = cfg["layer_params"]
        ws = cfg["world_size"]

        want_sharded = ref.calculate_sharded_elements(lp, ws)
        try:
            got_sharded = calculate_sharded_elements(lp, ws)
            if got_sharded == want_sharded:
                sharded_ok += 1
            elif "_note" not in out:
                out["_note"] = f"cfg {i}: sharded want {want_sharded}, got {got_sharded}"
        except Exception as e:
            if "_note" not in out: out["_note"] = f"sharded error: {e}"

        want_mem = ref.estimate_zero3_memory(lp, ws)
        try:
            got_mem = estimate_zero3_memory(lp, ws)
            if got_mem == want_mem:
                mem_ok += 1
            elif "_note" not in out:
                out["_note"] = f"cfg {i}: mem want {want_mem}, got {got_mem}"
        except Exception as e:
            if "_note" not in out: out["_note"] = f"mem error: {e}"

    out["sharded_exact"] = float(sharded_ok) / len(ref.CONFIGS)
    out["memory_exact"] = float(mem_ok) / len(ref.CONFIGS)
    return out
