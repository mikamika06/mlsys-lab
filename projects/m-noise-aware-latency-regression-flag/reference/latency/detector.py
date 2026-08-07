from latency.stats import compute_robust_stats

def classify_run(latencies, baseline_med, baseline_mad, max_rel_diff, reassociation_tol=1e-3):
    stats = compute_robust_stats(latencies)
    med = stats["median"]

    is_reassociation = max_rel_diff <= reassociation_tol
    noise_threshold = baseline_med + 5.0 * max(baseline_mad, 1.0)

    if med > noise_threshold:
        return "silent_eager_fallback"
    if is_reassociation and med <= noise_threshold:
        return "reassociation"
    return "normal"
