import ref


def check(workdir):
    from splitkv.occupancy import compute_split_count, partition_kv_ranges

    out = {"split_counts_matched": 0.0, "total_configs": float(len(ref.CONFIGS))}
    matched = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want_s = ref.compute_split_count(**cfg)
        got_s = compute_split_count(**cfg)
        want_r = ref.partition_kv_ranges(cfg["kv_len"], want_s)
        got_r = partition_kv_ranges(cfg["kv_len"], got_s)
        if got_s == want_s and got_r == want_r:
            matched += 1
        elif "_note" not in out:
            out["_note"] = f"cfg {i}: got split={got_s}, want split={want_s}"
    out["split_counts_matched"] = float(matched)
    return out
