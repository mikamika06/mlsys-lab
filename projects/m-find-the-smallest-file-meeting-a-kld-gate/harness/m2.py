import ref


def check(workdir):
    from quant.override import override_kv_config
    from quant.pipeline import run_pipeline

    out = {"overrides_matched": 0.0, "sizes_match": 0.0, "fallback_handled": 0.0}

    cfg = {"kv_config": {"head_dim": 32}}
    ov = ref.OVERRIDES_SET[0]
    got_ov = override_kv_config(cfg, ov)
    if got_ov and got_ov.get("kv_config", {}).get("head_dim") == ov["head_dim"]:
        out["overrides_matched"] = 1.0

    candidates = ref.CANDIDATES_SET[0]
    max_kld = ref.MAX_KLDS[0]
    got_pipe = run_pipeline(candidates, max_kld, ov)
    valid = [c for c in candidates if c["kld"] <= max_kld]
    ref_best = min(valid, key=lambda x: x["size"]) if valid else None
    if got_pipe and ref_best and got_pipe["name"] == ref_best["name"]:
        out["sizes_match"] = 1.0

    got_none = run_pipeline(candidates, 0.00001, ov)
    if got_none is None:
        out["fallback_handled"] = 1.0

    return out
