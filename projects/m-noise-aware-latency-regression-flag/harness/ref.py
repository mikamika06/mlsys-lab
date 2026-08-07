CONFIGS = [
    {
        "samples": [100, 102, 101, 103, 100, 102, 101, 102, 100, 101],
        "baseline_med": 101.0,
        "baseline_mad": 1.0,
        "max_rel_diff": 1e-4,
    },
    {
        "samples": [100, 101, 102, 350, 101, 100, 102, 101, 100, 101],
        "baseline_med": 101.0,
        "baseline_mad": 1.0,
        "max_rel_diff": 1e-3,
    },
    {
        "samples": [220, 225, 218, 222, 221, 219, 224, 220, 223, 221],
        "baseline_med": 100.0,
        "baseline_mad": 2.0,
        "max_rel_diff": 1e-5,
    },
]

def compute_robust_stats(latencies):
    if not latencies:
        return {"median": 0.0, "mad": 0.0, "p95": 0.0}
    s = sorted(latencies)
    n = len(s)
    if n % 2 == 1:
        med = float(s[n // 2])
    else:
        med = float(s[n // 2 - 1] + s[n // 2]) / 2.0

    devs = sorted([abs(x - med) for x in latencies])
    if n % 2 == 1:
        mad = float(devs[n // 2])
    else:
        mad = float(devs[n // 2 - 1] + devs[n // 2]) / 2.0

    p95_idx = int(0.95 * (n - 1))
    p95 = float(s[p95_idx])
    return {"median": med, "mad": mad, "p95": p95}

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

def validate_execution_mode(latencies, baseline_med, baseline_mad, max_rel_diff):
    status = classify_run(latencies, baseline_med, baseline_mad, max_rel_diff)
    if status == "silent_eager_fallback":
        return False
    return True
