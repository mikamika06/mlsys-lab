from sla.profiler import classify_sla_compliance


def compute_cost_efficiency(batch_profiles, target_sla, cost_per_cpu_second):
    """
    Computes QPS and cost per 1k requests for each batch size profile.
    """
    classification = classify_sla_compliance(batch_profiles, target_sla)
    metrics = {}

    for b, profile in batch_profiles.items():
        lats = profile["latencies"]
        total_requests = len(lats)
        total_cpu_sec = profile["cpu_time_sec"]

        qps = total_requests / total_cpu_sec if total_cpu_sec > 0 else 0.0
        cost_per_1k = (total_cpu_sec / total_requests) * 1000.0 * cost_per_cpu_second if total_requests > 0 else 0.0

        is_compliant = classification["results"][b]["compliant"]

        metrics[b] = {
            "qps": qps,
            "cost_per_1k_requests": cost_per_1k,
            "compliant": is_compliant,
        }

    return metrics


def recommend_optimal_batch_size(batch_profiles, target_sla, cost_per_cpu_second):
    """
    Selects the batch size that maximizes QPS among SLA-compliant profiles.
    Ties broken by lower cost_per_1k_requests, then smaller batch size.
    """
    metrics = compute_cost_efficiency(batch_profiles, target_sla, cost_per_cpu_second)
    compliant_batches = [b for b, m in metrics.items() if m["compliant"]]

    if not compliant_batches:
        return None

    best_batch = max(
        compliant_batches,
        key=lambda b: (
            metrics[b]["qps"],
            -metrics[b]["cost_per_1k_requests"],
            -b,
        ),
    )
    return best_batch
