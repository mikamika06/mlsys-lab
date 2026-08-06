def compute_speedup_and_overhead(acceptance_rate, draft_cost_ratio, target_cost):
    effective_steps = 1.0 + acceptance_rate
    baseline_cost = target_cost
    spec_cost = draft_cost_ratio * target_cost + (1.0 - acceptance_rate * 0.1) * target_cost
    speedup = baseline_cost / max(spec_cost, 1e-6)
    overhead = max(0.0, spec_cost - baseline_cost)
    net_helping = speedup > 1.0 and acceptance_rate > 0.4
    return {
        "speedup_ratio": float(speedup),
        "compute_overhead": float(overhead),
        "net_helping": bool(net_helping)
    }
