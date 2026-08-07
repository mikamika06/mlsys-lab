def requant_cost_share(nodes, total_cycles):
    if total_cycles <= 0:
        return 0.0
    requant_cycles = sum(n.get("cycles", 0) for n in nodes if n.get("type") == "requantize")
    return float(requant_cycles) / float(total_cycles)
