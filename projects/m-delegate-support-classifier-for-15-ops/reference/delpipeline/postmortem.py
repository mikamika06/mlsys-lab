def analyze_zero_node(graph, support_results):
    reasons = []
    for i, supported in enumerate(support_results):
        if not supported:
            op = graph["ops"][i]
            reasons.append(f"Op {op['name']} at index {i} is unsupported.")
    return {"delegated_nodes": 0, "reasons": reasons}
