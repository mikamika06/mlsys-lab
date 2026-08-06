import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from capacity.batching import find_optimal_batch_size
    from capacity.autoscaler import evaluate_autoscaler

    profiles = ref.generate_batch_profiles()
    sla_ms = 50.0
    cost_hr = 4.0
    
    want_batch = ref.ref_find_optimal_batch_size(profiles, sla_ms, cost_hr)
    try:
        got_batch = find_optimal_batch_size(profiles, sla_ms, cost_hr)
    except Exception as e:  # noqa: BLE001
        return {"batching_matched": 0.0, "cost_rel_err": 1.0, "_note": f"Exception in batching: {e}"}

    batch_ok = got_batch.get("optimal_batch_size") == want_batch["optimal_batch_size"]
    
    want_cost = want_batch["min_cost_per_1k_tokens"]
    got_cost = got_batch.get("min_cost_per_1k_tokens", 0.0)
    cost_err = abs(got_cost - want_cost) / abs(want_cost) if want_cost != 0 else 0.0

    arr_rate, srv_rate, trace = ref.generate_autoscaler_scenario()
    want_auto = ref.ref_evaluate_autoscaler(arr_rate, srv_rate, trace)
    try:
        got_auto = evaluate_autoscaler(arr_rate, srv_rate, trace)
    except Exception as e:  # noqa: BLE001
        return {"batching_matched": 0.0, "cost_rel_err": 1.0, "_note": f"Exception in autoscaler: {e}"}

    auto_ok = (got_auto.get("min_replicas") == want_auto["min_replicas"]) and (
        abs(got_auto.get("slack_ratio", 0.0) - want_auto["slack_ratio"]) < 1e-4
    )

    matched = 1.0 if (batch_ok and auto_ok and cost_err <= 0.001) else 0.0
    out = {"batching_matched": matched, "cost_rel_err": float(cost_err)}
    if not matched:
        out["_note"] = f"Batch result: got {got_batch}, want {want_batch}. Auto result: got {got_auto}, want {want_auto}"
    return out
