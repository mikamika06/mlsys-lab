import ref


def check(workdir):
    from sla.cost import compute_cost_efficiency, recommend_optimal_batch_size

    out = {"efficiency_matched": 0.0, "optimal_batch_matched": 0.0}

    want_metrics = ref.reference_cost_efficiency(ref.PROFILES, ref.TARGET_SLA, ref.COST_PER_CPU_SEC)
    want_optimal = ref.reference_recommend(ref.PROFILES, ref.TARGET_SLA, ref.COST_PER_CPU_SEC)

    try:
        got_metrics = compute_cost_efficiency(ref.PROFILES, ref.TARGET_SLA, ref.COST_PER_CPU_SEC)
    except Exception as e:
        out["_note"] = f"compute_cost_efficiency raised {type(e).__name__}: {e}"
        return out

    metrics_ok = True
    for b in ref.PROFILES:
        if b not in got_metrics:
            metrics_ok = False
            out["_note"] = f"Batch size {b} missing from cost efficiency result"
            break
        gm = got_metrics[b]
        wm = want_metrics[b]
        if gm.get("compliant") != wm["compliant"]:
            metrics_ok = False
            out["_note"] = f"Batch {b} compliance mismatch in cost efficiency"
            break
        if abs(gm.get("qps", 0.0) - wm["qps"]) > 1e-3:
            metrics_ok = False
            out["_note"] = f"Batch {b} QPS mismatch: got {gm.get('qps')}, want {wm['qps']}"
            break
        if abs(gm.get("cost_per_1k_requests", 0.0) - wm["cost_per_1k_requests"]) > 1e-5:
            metrics_ok = False
            out["_note"] = f"Batch {b} cost_per_1k mismatch"
            break

    if metrics_ok:
        out["efficiency_matched"] = 1.0

    try:
        got_optimal = recommend_optimal_batch_size(ref.PROFILES, ref.TARGET_SLA, ref.COST_PER_CPU_SEC)
    except Exception as e:
        out["_note"] = f"recommend_optimal_batch_size raised {type(e).__name__}: {e}"
        return out

    if got_optimal == want_optimal:
        out["optimal_batch_matched"] = 1.0
    else:
        out["_note"] = f"optimal batch mismatch: got {got_optimal}, want {want_optimal}"

    return out
