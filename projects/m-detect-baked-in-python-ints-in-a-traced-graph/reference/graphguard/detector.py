def detect_baked_ints(graph):
    """Detect baked in python ints."""
    out = []
    for node in graph["nodes"]:
        if node["op"] == "const" and isinstance(node.get("val"), int):
            out.append(node["id"])
    return sorted(out)
