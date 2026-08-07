import sys

sys.path.insert(0, ".")
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from profile_parser.parser import parse_ane_power_mw
    from profile_parser.telemetry import correlate_residency_with_throughput, estimate_ane_utilization

    text, _, want_ane = ref.generate_powermetrics_fixture()
    got_ane = parse_ane_power_mw(text)

    out = {
        "correlation_rel_err": 1.0,
        "ane_power_rel_err": 1.0,
        "ane_util_rel_err": 1.0,
    }

    if len(got_ane) != len(want_ane):
        out["_note"] = f"Expected {len(want_ane)} ANE samples, got {len(got_ane)}"
        return out

    ane_errs = [abs(g - w) / (abs(w) + 1e-9) for g, w in zip(got_ane, want_ane)]
    out["ane_power_rel_err"] = float(max(ane_errs))

    all_gpu = []
    all_tps = []
    for model_key in ["model_3b", "model_7b", "model_13b"]:
        run = ref.MODEL_RUNS[model_key]
        all_gpu.extend(run["gpu_residency"])
        all_tps.extend(run["tps"])

    want_corr = ref.reference_correlate(all_gpu, all_tps)
    got_corr = correlate_residency_with_throughput(all_gpu, all_tps)

    c_err = abs(got_corr.get("correlation", 0.0) - want_corr["correlation"]) / (abs(want_corr["correlation"]) + 1e-9)
    out["correlation_rel_err"] = float(c_err)

    want_util = ref.reference_ane_utilization(got_ane, max_power_mw=8000.0)
    got_util = estimate_ane_utilization(got_ane, max_ane_power_mw=8000.0)

    u_err = abs(got_util.get("estimated_utilization_pct", 0.0) - want_util["estimated_utilization_pct"]) / (
        abs(want_util["estimated_utilization_pct"]) + 1e-9
    )
    out["ane_util_rel_err"] = float(u_err)

    return out
