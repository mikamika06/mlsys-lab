import ref


def check(workdir):
    from meshshard.traffic import compute_traffic

    out = {"traffic_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        want = ref.compute_traffic(cfg["mesh_shape"], cfg["shard_config"])
        got = compute_traffic(cfg["mesh_shape"], cfg["shard_config"])
        if got == want:
            ok += 1
    out["traffic_matched"] = float(ok)
    return out
