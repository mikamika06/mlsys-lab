def parse_engine_inspector_log(inspector_data):
    layers = []
    items = inspector_data.get("bindings", []) + inspector_data.get("layers", [])
    for item in items:
        if "Name" in item:
            layers.append({
                "name": item["Name"],
                "type": item.get("LayerType", "Unknown"),
                "tactic": item.get("TacticName", "default"),
                "precision": item.get("Precision", "FP32"),
            })
    return layers


def diff_engine_graphs(graph_a, graph_b):
    map_a = {l["name"]: l for l in graph_a}
    map_b = {l["name"]: l for l in graph_b}

    names_a = set(map_a.keys())
    names_b = set(map_b.keys())

    added = list(names_b - names_a)
    removed = list(names_a - names_b)
    tactic_mismatches = []
    precision_mismatches = []

    common = names_a.intersection(names_b)
    for name in common:
        la = map_a[name]
        lb = map_b[name]
        if la.get("tactic") != lb.get("tactic"):
            tactic_mismatches.append({
                "name": name,
                "tactic_a": la.get("tactic"),
                "tactic_b": lb.get("tactic"),
            })
        if la.get("precision") != lb.get("precision"):
            precision_mismatches.append({
                "name": name,
                "precision_a": la.get("precision"),
                "precision_b": lb.get("precision"),
            })

    return {
        "added": sorted(added),
        "removed": sorted(removed),
        "tactic_mismatches": sorted(tactic_mismatches, key=lambda x: x["name"]),
        "precision_mismatches": sorted(precision_mismatches, key=lambda x: x["name"]),
        "is_identical": len(added) == 0 and len(removed) == 0 and len(tactic_mismatches) == 0 and len(precision_mismatches) == 0,
    }
