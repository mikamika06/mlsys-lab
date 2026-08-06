def analyze_fallback_vs_rewrite(nodes, runtime_metrics):
    decisions = {}
    total_estimated_latency = 0.0

    for node in nodes:
        node_id = node["id"]
        op_type = node["op_type"]
        metrics = runtime_metrics.get(op_type, {})

        fallback_latency = metrics.get("custom_op_overhead", 10.0) + metrics.get("fallback_exec", 5.0)
        rewrite_latency = metrics.get("decomposition_exec", 8.0)

        if rewrite_latency <= fallback_latency:
            choice = "REWRITE"
            lat = rewrite_latency
        else:
            choice = "FALLBACK"
            lat = fallback_latency

        decisions[node_id] = {
            "strategy": choice,
            "latency": lat
        }
        total_estimated_latency += lat

    return {
        "decisions": decisions,
        "total_latency": total_estimated_latency
    }
