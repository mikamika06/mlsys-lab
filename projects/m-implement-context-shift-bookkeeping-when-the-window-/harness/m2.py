import ref


def check(workdir):
    from ctxshift import kv_cache_bytes, mha_vs_gqa

    out = {"bytes_match": 1.0, "ratio_match": 1.0, "savings_nonneg": 1.0, "mha_is_noop": 1.0}
    notes = []
    for i, cfg in enumerate(ref.CONFIGS):
        want_bytes = ref.kv_cache_bytes(cfg)
        got_bytes = kv_cache_bytes(dict(cfg))
        if got_bytes != want_bytes:
            out["bytes_match"] = 0.0
            notes.append(f"config {i}: kv_cache_bytes got {got_bytes}, reference {want_bytes}")

        got_cmp = mha_vs_gqa(dict(cfg))
        got_mha = got_cmp.get("mha_bytes") if isinstance(got_cmp, dict) else None
        got_gqa = got_cmp.get("gqa_bytes") if isinstance(got_cmp, dict) else None
        got_saved = got_cmp.get("saved_bytes") if isinstance(got_cmp, dict) else None

        if got_mha is None or got_gqa is None or got_gqa * cfg["n_heads"] != got_mha * cfg["n_kv_heads"]:
            out["ratio_match"] = 0.0
            notes.append(f"config {i}: ratio mismatch mha={got_mha} gqa={got_gqa}")

        if got_saved is None or got_saved < 0:
            out["savings_nonneg"] = 0.0

        if cfg["n_kv_heads"] == cfg["n_heads"] and (got_saved or 0) != 0:
            out["mha_is_noop"] = 0.0

    if notes:
        out["_note"] = " | ".join(notes[:2])
    return out
