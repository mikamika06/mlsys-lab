def count_aten_ops(graph_data):
    counts = {}
    for node in graph_data.get("nodes", []):
        if node.get("target_type") == "aten":
            name = node.get("name")
            counts[name] = counts.get(name, 0) + 1
    return counts


def identify_decompositions(graph_data, target_ops):
    decomposed = {}
    for op in target_ops:
        found = False
        for node in graph_data.get("nodes", []):
            if node.get("target_type") == "aten" and node.get("name") == op:
                if node.get("decomposed", False):
                    found = True
                    break
        decomposed[op] = found
    return decomposed
