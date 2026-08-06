import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from specdiag.goodput import aggregate_goodput_comparison

    log = ref.make_scheduler_logs()

    import specdiag.goodput as ref_gp
    ref_res = ref_gp.aggregate_goodput_comparison(log, penalty_factor=0.5)

    out = {"throughput_ratio": 0.0, "metrics_match": 0.0}
    try:
        user_res = aggregate_goodput_comparison(log, penalty_factor=0.5)

        out["throughput_ratio"] = float(user_res.get("throughput_ratio", 0.0))

        match_spec = abs(user_res.get("spec_aggregate_goodput_tps", 0.0) - ref_res["spec_aggregate_goodput_tps"]) < 1e-4
        match_base = abs(user_res.get("base_aggregate_goodput_tps", 0.0) - ref_res["base_aggregate_goodput_tps"]) < 1e-4
        match_ratio = abs(user_res.get("throughput_ratio", 0.0) - ref_res["throughput_ratio"]) < 1e-4

        if match_spec and match_base and match_ratio:
            out["metrics_match"] = 1.0
        else:
            out["_note"] = f"Reference metrics {ref_res}, got {user_res}"
    except Exception as e:  # noqa: BLE001
        out["_note"] = f"Goodput aggregation failed: {type(e).__name__}: {str(e)[:120]}"

    return out
