def attribute_size_regression(base_tensors, candidate_tensors):
    """Attribute size changes between base and candidate weight manifests."""
    base_map = {t["name"]: t for t in base_tensors}
    cand_map = {t["name"]: t for t in candidate_tensors}

    base_total = sum(t["size_bytes"] for t in base_tensors)
    cand_total = sum(t["size_bytes"] for t in candidate_tensors)
    net_delta = cand_total - base_total

    all_layers = set()
    for t in base_tensors:
        all_layers.add(t.get("layer", "default"))
    for t in candidate_tensors:
        all_layers.add(t.get("layer", "default"))

    by_layer = {layer: 0 for layer in sorted(all_layers)}
    for t in candidate_tensors:
        by_layer[t.get("layer", "default")] += t["size_bytes"]
    for t in base_tensors:
        by_layer[t.get("layer", "default")] -= t["size_bytes"]

    added_delta = 0
    removed_delta = 0
    modified_delta = 0

    all_names = set(base_map.keys()) | set(cand_map.keys())
    contributors = []

    for name in all_names:
        in_base = name in base_map
        in_cand = name in cand_map

        if in_cand and not in_base:
            c_t = cand_map[name]
            delta = c_t["size_bytes"]
            added_delta += delta
            contributors.append({
                "name": name,
                "layer": c_t.get("layer", "default"),
                "delta_bytes": delta,
            })
        elif in_base and not in_cand:
            b_t = base_map[name]
            delta = -b_t["size_bytes"]
            removed_delta += delta
            contributors.append({
                "name": name,
                "layer": b_t.get("layer", "default"),
                "delta_bytes": delta,
            })
        else:
            b_t = base_map[name]
            c_t = cand_map[name]
            delta = c_t["size_bytes"] - b_t["size_bytes"]
            if delta != 0:
                modified_delta += delta
                contributors.append({
                    "name": name,
                    "layer": c_t.get("layer", "default"),
                    "delta_bytes": delta,
                })

    contributors.sort(key=lambda x: (-abs(x["delta_bytes"]), x["name"]))

    return {
        "total_base_bytes": base_total,
        "total_candidate_bytes": cand_total,
        "net_delta_bytes": net_delta,
        "by_layer": by_layer,
        "category_deltas": {
            "added": added_delta,
            "removed": removed_delta,
            "modified": modified_delta,
        },
        "top_contributors": contributors,
    }
