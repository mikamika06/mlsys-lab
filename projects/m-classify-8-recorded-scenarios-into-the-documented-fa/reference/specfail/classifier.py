TAXONOMY = [
    "draft_acceptance_collapse",
    "verifier_divergence_mismatch",
    "speculative_cache_corruption",
    "rollback_index_out_of_bounds",
    "stream_synchronization_stall",
    "memory_leak_speculative_buffer",
    "numerical_drift_concurrency",
    "cascading_latency_penalty"
]

def classify_scenarios(scenarios):
    results = []
    for s in scenarios:
        sid = s.get("id")
        features = s.get("features", {})
        accept_rate = features.get("accept_rate", 1.0)
        logit_diff = features.get("logit_diff", 0.0)
        cache_err = features.get("cache_err", 0)
        rollback_fault = features.get("rollback_fault", 0)
        sync_stall = features.get("sync_stall", 0)
        mem_leak = features.get("mem_leak", 0)
        num_drift = features.get("num_drift", 0.0)
        latency_spike = features.get("latency_spike", 0.0)

        if accept_rate < 0.05:
            cat = "draft_acceptance_collapse"
        elif logit_diff > 5.0:
            cat = "verifier_divergence_mismatch"
        elif cache_err > 0:
            cat = "speculative_cache_corruption"
        elif rollback_fault > 0:
            cat = "rollback_index_out_of_bounds"
        elif sync_stall > 0:
            cat = "stream_synchronization_stall"
        elif mem_leak > 0:
            cat = "memory_leak_speculative_buffer"
        elif num_drift > 1e-3:
            cat = "numerical_drift_concurrency"
        elif latency_spike > 2.0:
            cat = "cascading_latency_penalty"
        else:
            cat = "draft_acceptance_collapse"
        results.append({"id": sid, "category": cat})
    return results
