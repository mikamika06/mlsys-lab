import ref


def check(workdir):
    from bwbound.analytic import compute_analytic_bound
    from bwbound.measured import analyze_measured_performance

    max_rel_err = 0.0
    matched = 0

    for w in ref.WORKLOADS:
        analytic = compute_analytic_bound(
            w["tensors"], w["flops"], w["peak_bandwidth_gbps"], w["peak_tflops"]
        )
        total_bytes = analytic["total_bytes"]
        want = ref.reference_analyze_measured(
            w["measured_time_sec"], total_bytes, w["flops"], w["peak_bandwidth_gbps"], w["peak_tflops"]
        )
        try:
            got = analyze_measured_performance(
                w["measured_time_sec"], total_bytes, w["flops"], w["peak_bandwidth_gbps"], w["peak_tflops"]
            )
        except Exception as e:
            return {
                "m2_rel_err": 1.0,
                "m2_matched": 0.0,
                "_note": f"Execution failed: {type(e).__name__}: {str(e)[:120]}",
            }

        for key in ["achieved_gbps", "achieved_tflops", "bandwidth_utilization", "compute_utilization", "rel_err", "efficiency_ratio"]:
            err = abs(float(got[key]) - float(want[key])) / max(abs(float(want[key])), 1e-12)
            if err > max_rel_err:
                max_rel_err = err

        matched += 1

    matched_ratio = matched / float(len(ref.WORKLOADS))
    return {"m2_rel_err": max_rel_err, "m2_matched": matched_ratio}
