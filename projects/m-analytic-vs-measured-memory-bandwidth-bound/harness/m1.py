import ref


def check(workdir):
    from bwbound.analytic import compute_analytic_bound

    max_rel_err = 0.0
    matched = 0

    for w in ref.WORKLOADS:
        want = ref.reference_compute_analytic(
            w["tensors"], w["flops"], w["peak_bandwidth_gbps"], w["peak_tflops"]
        )
        try:
            got = compute_analytic_bound(
                w["tensors"], w["flops"], w["peak_bandwidth_gbps"], w["peak_tflops"]
            )
        except Exception as e:
            return {
                "m1_rel_err": 1.0,
                "m1_matched": 0.0,
                "_note": f"Execution failed: {type(e).__name__}: {str(e)[:120]}",
            }

        for key in ["total_bytes", "arithmetic_intensity", "time_mem_sec", "time_compute_sec", "analytic_time_sec"]:
            err = abs(float(got[key]) - float(want[key])) / max(abs(float(want[key])), 1e-12)
            if err > max_rel_err:
                max_rel_err = err

        if got.get("is_memory_bound") == want["is_memory_bound"]:
            matched += 1

    matched_ratio = matched / float(len(ref.WORKLOADS))
    return {"m1_rel_err": max_rel_err, "m1_matched": matched_ratio}
