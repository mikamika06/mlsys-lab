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

def get_scenarios():
    scenarios = []
    for i, cat in enumerate(TAXONOMY):
        features = {
            "accept_rate": 0.001 if cat == "draft_acceptance_collapse" else 0.9,
            "logit_diff": 10.0 if cat == "verifier_divergence_mismatch" else 0.1,
            "cache_err": 1 if cat == "speculative_cache_corruption" else 0,
            "rollback_fault": 1 if cat == "rollback_index_out_of_bounds" else 0,
            "sync_stall": 1 if cat == "stream_synchronization_stall" else 0,
            "mem_leak": 1 if cat == "memory_leak_speculative_buffer" else 0,
            "num_drift": 0.1 if cat == "numerical_drift_concurrency" else 0.0,
            "latency_spike": 5.0 if cat == "cascading_latency_penalty" else 0.0
        }
        scenarios.append({"id": f"scenario_{i}", "features": features})
    return scenarios

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

def compute_metrics(classifications):
    counts = {t: 0 for t in TAXONOMY}
    for c in classifications:
        cat = c.get("category")
        if cat in counts:
            counts[cat] += 1
    total = len(classifications)
    distribution = {k: (v / total if total > 0 else 0.0) for k, v in counts.items()}
    confidence_score = sum(1.0 for v in counts.values() if v > 0) / len(TAXONOMY)
    return {
        "counts": counts,
        "distribution": distribution,
        "confidence_score": confidence_score
    }
