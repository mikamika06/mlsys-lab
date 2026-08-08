def measure_speedup_gap(dense_profile, pruned_profile):
    dense_time = dense_profile.get("latency_ms", 100.0)
    pruned_time = pruned_profile.get("latency_ms", 80.0)
    theoretical_sparsity = pruned_profile.get("sparsity_ratio", 0.5)
    real_speedup = dense_time / max(1e-5, pruned_time)
    theoretical_speedup = 1.0 / max(1e-5, (1.0 - theoretical_sparsity))
    gap = abs(real_speedup - theoretical_speedup)
    return {
        "real_speedup": real_speedup,
        "theoretical_speedup": theoretical_speedup,
        "gap": gap
    }
