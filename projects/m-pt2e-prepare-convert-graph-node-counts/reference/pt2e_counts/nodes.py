def classify_node(node):
    op = node.get("op")
    target = node.get("target", "")
    if op == "call_module":
        if "observer" in str(target).lower() or "quant" in str(target).lower():
            return "observer_quant"
        return "call_module"
    return op or "other"


def extract_node_stats(graph):
    stats = {}
    for node in graph.get("nodes", []):
        t = classify_node(node)
        stats[t] = stats.get(t, 0) + 1
    stats["total"] = len(graph.get("nodes", []))
    return stats
