import sys


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    import ref

    try:
        from blockalign.planner import select_optimal_block_size
    except Exception as e:
        return {
            "optimal_selected": 0.0,
            "memory_waste_matched": 0.0,
            "_note": f"Failed to import planner module: {type(e).__name__}: {e}",
        }

    out = {
        "optimal_selected": 0.0,
        "memory_waste_matched": 0.0,
        "total_configs": float(len(ref.CONFIGS_M2)),
    }

    opt_ok = 0
    waste_ok = 0

    for i, cfg in enumerate(ref.CONFIGS_M2):
        backend = cfg["backend"]
        model = cfg["model"]
        candidates = cfg["candidates"]
        max_mem = cfg["max_memory_bytes"]
        prompt_lens = cfg["prompt_lens"]

        want = ref.select_optimal_block_size(backend, model, candidates, max_mem, prompt_lens)
        try:
            got = select_optimal_block_size(backend, model, candidates, max_mem, prompt_lens)
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"M2 config {i} select_optimal_block_size raised: {type(e).__name__}: {e}"
            continue

        if got.get("best_block_size") == want.get("best_block_size"):
            opt_ok += 1
        elif "_note" not in out:
            out["_note"] = f"M2 config {i} best_bs mismatch: got {got.get('best_block_size')}, want {want.get('best_block_size')}"

        if (
            got.get("allocated_bytes") == want.get("allocated_bytes")
            and got.get("fragmentation_bytes") == want.get("fragmentation_bytes")
            and got.get("valid_count") == want.get("valid_count")
        ):
            waste_ok += 1
        elif "_note" not in out:
            out["_note"] = f"M2 config {i} metrics mismatch: got {got}, want {want}"

    if opt_ok == len(ref.CONFIGS_M2):
        out["optimal_selected"] = 1.0
    if waste_ok == len(ref.CONFIGS_M2):
        out["memory_waste_matched"] = 1.0

    return out
