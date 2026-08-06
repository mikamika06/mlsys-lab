import ref

def check(workdir):
    from autotp.sharding import shard_kv_heads
    from autotp.search import find_tp_sweet_spot

    out = {"sharding_match": 0.0, "sweet_spot_match": 0.0}

    s_want = ref.shard_kv_heads(8, 3)
    s_got = shard_kv_heads(8, 3)
    if s_want == s_got:
        out["sharding_match"] = 1.0
    else:
        out["_note"] = f"sharding mismatch: got {s_got}, want {s_want}"

    cfg = ref.CONFIGS[0]
    hw = ref.HW_PROFILES[0]
    w_tp = ref.find_tp_sweet_spot(cfg, hw, 1)
    g_tp = find_tp_sweet_spot(cfg, hw, 1)
    if w_tp == g_tp:
        out["sweet_spot_match"] = 1.0
    else:
        if "_note" not in out:
            out["_note"] = f"sweet spot mismatch: got {g_tp}, want {w_tp}"

    return out
