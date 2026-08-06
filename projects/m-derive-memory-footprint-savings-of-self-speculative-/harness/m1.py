import ref


def check(workdir):
    from speculative.memory import compute_weight_memory, compute_kv_cache_memory
    out = {"footprints_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, (target_cfg, draft_cfg, bs, sl) in enumerate(ref.CONFIGS):
        ref_w = ref.compute_weight_memory(target_cfg)
        ref_kv = ref.compute_kv_cache_memory(target_cfg, bs, sl)
        try:
            got_w = compute_weight_memory(target_cfg)
            got_kv = compute_kv_cache_memory(target_cfg, bs, sl)
            if got_w == ref_w and got_kv == ref_kv:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"config {i}: got weight={got_w}, kv={got_kv}; want weight={ref_w}, kv={ref_kv}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"config {i} raised error: {type(e).__name__}: {str(e)[:100]}"
    out["footprints_matched"] = float(ok)
    return out
