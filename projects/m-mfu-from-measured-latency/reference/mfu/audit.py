def audit_speedup(baseline_latency, optimized_latency, claimed_speedup):
    actual_speedup = baseline_latency / optimized_latency
    discrepancy = abs(actual_speedup - claimed_speedup) / claimed_speedup
    return {
        "actual_speedup": float(actual_speedup),
        "claimed_speedup": float(claimed_speedup),
        "discrepancy": float(discrepancy),
        "is_consistent": bool(discrepancy < 0.05)
    }
