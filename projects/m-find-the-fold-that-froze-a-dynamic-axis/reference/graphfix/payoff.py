def estimate_simplification_payoff(graph_before, graph_after):
    nodes_before = len(graph_before.get("nodes", []))
    nodes_after = len(graph_after.get("nodes", []))
    init_before = len(graph_before.get("initializers", {}))
    init_after = len(graph_after.get("initializers", {}))

    node_reduction = (nodes_before - nodes_after) / max(1, nodes_before)
    init_reduction = (init_before - init_after) / max(1, init_before)
    estimated_latency_gain = node_reduction * 0.15 + init_reduction * 0.05
    return {
        "nodes_removed": nodes_before - nodes_after,
        "initializers_removed": init_before - init_after,
        "latency_payoff_ratio": float(estimated_latency_gain)
    }
