def build_configs(config):
    buckets = {}
    for sub in config["submodules"]:
        key = (sub["kind"], sub["num_heads"], sub["kv_heads"], sub["head_dim"], sub["causal"])
        buckets.setdefault(key, []).append(sub["index"])
    out = []
    for (kind, nh, kv, hd, causal), indices in sorted(buckets.items(), key=lambda x: min(x[1])):
        out.append({
            "kind": kind,
            "num_heads": nh,
            "kv_heads": kv,
            "head_dim": hd,
            "causal": causal,
            "submodules": sorted(indices)
        })
    return out
