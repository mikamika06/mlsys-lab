def measure_coverage(routing_data, num_experts):
    """Measure expert coverage from routing activations."""
    counts = {i: 0 for i in range(num_experts)}
    total_tokens = len(routing_data)
    for experts in routing_data:
        for e in set(experts):
            if 0 <= e < num_experts:
                counts[e] += 1
    coverage_ratio = sum(1 for e, c in counts.items() if c > 0) / num_experts
    return {"counts": counts, "total_tokens": total_tokens, "coverage_ratio": coverage_ratio}
