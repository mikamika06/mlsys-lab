def analyze_path(model_spec):
    nodes = model_spec.get("nodes", [])
    decomposed_cost = sum(n.get("flops", 0) * 1.5 for n in nodes)
    plugin_cost = sum(n.get("flops", 0) * 0.8 for n in nodes) + 1000
    overhead_diff = abs(decomposed_cost - plugin_cost)
    recommendation = "plugin" if plugin_cost < decomposed_cost else "decomposed"
    return {
        "decomposed_cost": float(decomposed_cost),
        "plugin_cost": float(plugin_cost),
        "overhead_diff": float(overhead_diff),
        "recommendation": recommendation,
    }
