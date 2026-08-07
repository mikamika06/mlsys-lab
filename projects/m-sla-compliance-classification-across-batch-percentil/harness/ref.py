import numpy as np


def generate_batch_profiles():
    np.random.seed(42)
    profiles = {}
    batch_sizes = [1, 2, 4, 8, 16]

    for b in batch_sizes:
        n_req = 200
        base_lat = 5.0 + 1.2 * b
        tail_noise = np.random.pareto(a=2.0, size=n_req) * (b * 1.5)
        latencies = (base_lat + tail_noise).tolist()
        cpu_time = (sum(latencies) / 1000.0) * 0.8
        profiles[b] = {
            "latencies": latencies,
            "cpu_time_sec": float(cpu_time),
        }

    return profiles


PROFILES = generate_batch_profiles()
TARGET_SLA = {50.0: 12.0, 90.0: 20.0, 95.0: 25.0, 99.0: 35.0}
COST_PER_CPU_SEC = 0.00005


def reference_classify(batch_profiles, target_sla):
    results = {}
    max_compliant = None
    sorted_batches = sorted(batch_profiles.keys())
    target_pcts = sorted(list(target_sla.keys()))

    for b in sorted_batches:
        profile = batch_profiles[b]
        lats = profile["latencies"]
        arr = np.array(lats, dtype=np.float64)

        measured_pcts = {}
        for p in target_pcts:
            measured_pcts[p] = float(np.percentile(arr, p))

        violations = []
        for p in target_pcts:
            if measured_pcts[p] > target_sla[p]:
                violations.append(p)

        is_compliant = len(violations) == 0
        results[b] = {
            "compliant": is_compliant,
            "percentiles": measured_pcts,
            "violations": violations,
        }

        if is_compliant:
            if max_compliant is None or b > max_compliant:
                max_compliant = b

    return {
        "results": results,
        "max_compliant_batch": max_compliant,
    }


def reference_cost_efficiency(batch_profiles, target_sla, cost_per_cpu_second):
    classification = reference_classify(batch_profiles, target_sla)
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


def reference_recommend(batch_profiles, target_sla, cost_per_cpu_second):
    metrics = reference_cost_efficiency(batch_profiles, target_sla, cost_per_cpu_second)
    compliant_batches = [b for b, m in metrics.items() if m["compliant"]]

    if not compliant_batches:
        return None

    return max(
        compliant_batches,
        key=lambda b: (
            metrics[b]["qps"],
            -metrics[b]["cost_per_1k_requests"],
            -b,
        ),
    )
