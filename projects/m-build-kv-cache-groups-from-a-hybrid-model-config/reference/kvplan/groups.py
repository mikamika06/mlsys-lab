def build_groups(config):
    buckets = {}
    for layer in config["layers"]:
        key = (layer["kind"], layer.get("window") or 0, layer["kv_heads"], layer["head_dim"])
        buckets.setdefault(key, []).append(layer["index"])
    groups = []
    for (kind, window, kv_heads, head_dim), idx in sorted(buckets.items()):
        groups.append({"kind": kind, "window": window, "kv_heads": kv_heads,
                       "head_dim": head_dim, "layers": sorted(idx)})
    return groups
