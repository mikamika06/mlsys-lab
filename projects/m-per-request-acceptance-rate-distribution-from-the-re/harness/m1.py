import ref


def check(workdir):
    from eagle_diag.metrics import compute_distribution_summary, compute_request_acceptance

    out = {"acceptance_rate": 0.0}
    want_stats = ref.reference_compute_request_acceptance(ref.RECORDED_EAGLE_LOGS)
    want_dist = ref.reference_compute_distribution_summary(want_stats)

    try:
        got_stats = compute_request_acceptance(ref.RECORDED_EAGLE_LOGS)
        got_dist = compute_distribution_summary(got_stats)
    except Exception as e:
        out["_note"] = f"Execution error: {type(e).__name__}: {str(e)[:120]}"
        return out

    if len(got_stats) != len(want_stats):
        out["_note"] = f"Expected {len(want_stats)} request stats, got {len(got_stats)}"
        return out

    stats_ok = True
    for g, w in zip(got_stats, want_stats):
        if g["request_id"] != w["request_id"]:
            stats_ok = False
            break
        if g["total_accepted"] != w["total_accepted"]:
            stats_ok = False
            break
        if g["total_proposed"] != w["total_proposed"]:
            stats_ok = False
            break
        if abs(g["mean_acceptance_rate"] - w["mean_acceptance_rate"]) > 1e-6:
            stats_ok = False
            break

    if not stats_ok:
        out["_note"] = f"Request stats mismatch. Got {got_stats[:1]}, want {want_stats[:1]}"
        return out

    dist_ok = True
    for key in ("p25", "p50", "p75", "mean"):
        if abs(got_dist.get(key, 0.0) - want_dist[key]) > 1e-5:
            dist_ok = False
            break

    if not dist_ok:
        out["_note"] = f"Distribution summary mismatch. Got {got_dist}, want {want_dist}"
        return out

    out["acceptance_rate"] = 1.0
    return out
