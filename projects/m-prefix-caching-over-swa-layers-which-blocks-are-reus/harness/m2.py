import ref


def check(workdir):
    from prefixcache.hybrid import compute_mamba_state_size
    from prefixcache.cost import compute_dense_vs_hybrid_cost

    out = {"state_match": 0.0, "size_ratio": 0.0}
    state_ok = 0
    ratio_ok = 0

    for cfg in ref.CONFIGS:
        for layer in cfg["layers"]:
            if layer["kind"] == "mamba":
                want_s = ref.compute_mamba_state_size(layer)
                got_s = compute_mamba_state_size(layer)
                if got_s == want_s:
                    state_ok += 1

        costs_ref = ref.compute_dense_vs_hybrid_cost(cfg, 131072)
        costs_got = compute_dense_vs_hybrid_cost(cfg, 131072)
        if costs_ref["hybrid_bytes"] > 0:
            ratio_ref = costs_ref["dense_bytes"] / costs_ref["hybrid_bytes"]
            ratio_got = costs_got["dense_bytes"] / costs_got["hybrid_bytes"]
            if abs(ratio_ref - ratio_got) < 1e-5:
                ratio_ok += 1

    out["state_match"] = 1.0 if state_ok > 0 else 0.0
    out["size_ratio"] = 1.0 if ratio_ok >= len(ref.CONFIGS) else 0.0
    return out
