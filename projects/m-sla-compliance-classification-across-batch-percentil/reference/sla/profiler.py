import numpy as np


def calculate_percentiles(latencies, percentiles):
    """Calculates exact empirical percentiles using nearest-rank / linear interpolation."""
    arr = np.array(latencies, dtype=np.float64)
    res = {}
    for p in percentiles:
        res[p] = float(np.percentile(arr, p))
    return res


def classify_sla_compliance(batch_profiles, target_sla):
    """
    Evaluates SLA compliance for each batch size profile.

    batch_profiles: dict mapping batch_size (int) to dict with keys:
        - "latencies": list of float (ms)
        - "cpu_time_sec": float

    target_sla: dict mapping percentile (float) to max_latency_ms (float), e.g. {50.0: 10.0, 95.0: 25.0, 99.0: 50.0}

    Returns dict:
        - "results": dict mapping batch_size to dict:
            - "compliant": bool
            - "percentiles": dict mapping percentile to float
            - "violations": list of percentiles where measured > max_allowed
        - "max_compliant_batch": int or None
    """
    results = {}
    max_compliant = None

    sorted_batches = sorted(batch_profiles.keys())
    target_pcts = sorted(list(target_sla.keys()))

    for b in sorted_batches:
        profile = batch_profiles[b]
        lats = profile["latencies"]
        measured_pcts = calculate_percentiles(lats, target_pcts)

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
