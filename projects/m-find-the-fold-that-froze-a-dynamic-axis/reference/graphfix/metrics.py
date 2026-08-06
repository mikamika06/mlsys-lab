def compute_simplification_payoff(graph_before, graph_after):
    nodes_before = len(graph_before.get("nodes", []))
    nodes_after = len(graph_after.get("nodes", []))
    init_before = len(graph_before.get("initializers", {}))
    init_after = len(graph_after.get("initializers", {}))

    nodes_removed = nodes_before - nodes_after
    inits_removed = init_before - init_after

    node_reduction_pct = (nodes_removed / nodes_before) * 100.0 if nodes_before > 0 else 0.0
    init_reduction_pct = (inits_removed / init_before) * 100.0 if init_before > 0 else 0.0

    bytes_before = sum(len(v) for v in graph_before.get("initializers", {}).values())
    bytes_after = sum(len(v) for v in graph_after.get("initializers", {}).values())
    bytes_saved = bytes_before - bytes_after

    latency_payoff_estimate_ms = round(nodes_removed * 0.12 + (bytes_saved / 1024.0) * 0.05, 3)

    return {
        "nodes_removed": nodes_removed,
        "initializers_removed": inits_removed,
        "node_reduction_pct": round(node_reduction_pct, 2),
        "init_reduction_pct": round(init_reduction_pct, 2),
        "bytes_saved": bytes_saved,
        "latency_payoff_ms": latency_payoff_estimate_ms
    }
