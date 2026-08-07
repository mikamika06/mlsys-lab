def extract_submodule_groups(config):
    submodules = config.get("submodules", {})
    out = []
    for sub_name, sub_cfg in sorted(submodules.items()):
        layers = sub_cfg.get("layers", [])
        buckets = {}
        for layer in layers:
            key = (sub_name, layer.get("kind", "full"), layer.get("window"), layer.get("kv_heads", 1), layer.get("head_dim", 64))
            buckets.setdefault(key, []).append(layer["index"])
        for (sname, kind, window, kv_heads, head_dim), idxs in sorted(buckets.items()):
            group = {
                "submodule": sname,
                "kind": kind,
                "kv_heads": kv_heads,
                "head_dim": head_dim,
                "layers": sorted(idxs)
            }
            if window is not None:
                group["window"] = window
            out.append(group)
    return out
