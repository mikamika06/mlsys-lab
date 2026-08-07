import numpy as np


def compute_request_acceptance(log_records):
    results = []
    for rec in log_records:
        req_id = rec["request_id"]
        draft_accepted = rec["draft_accepted_counts"]
        draft_proposed = rec["draft_proposed_counts"]
        total_accepted = sum(draft_accepted)
        total_proposed = sum(draft_proposed)
        rate = float(total_accepted / total_proposed) if total_proposed > 0 else 0.0
        results.append({
            "request_id": req_id,
            "total_accepted": total_accepted,
            "total_proposed": total_proposed,
            "mean_acceptance_rate": rate,
        })
    return results


def compute_distribution_summary(request_stats):
    rates = [r["mean_acceptance_rate"] for r in request_stats]
    if not rates:
        return {"p25": 0.0, "p50": 0.0, "p75": 0.0, "mean": 0.0}
    arr = np.array(rates, dtype=np.float64)
    return {
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "mean": float(np.mean(arr)),
    }
