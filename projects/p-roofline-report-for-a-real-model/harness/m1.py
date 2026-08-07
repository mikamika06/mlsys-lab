def check(workdir):
    import sys
    import ref

    sys.path.insert(0, workdir)
    import roofline.intensity as intensity_mod

    m = {"intensity_math_ok": 0.0, "aggregation_ok": 0.0, "totals_ok": 0.0}

    try:
        val1 = intensity_mod.compute_kernel_intensity(1000.0, 250.0)
        val0 = intensity_mod.compute_kernel_intensity(1000.0, 0.0)
        if abs(val1 - 4.0) < 1e-6 and val0 == 0.0:
            m["intensity_math_ok"] = 1.0
    except Exception:
        return m

    raw = ref.generate_raw_profile(42)
    try:
        learner_agg = intensity_mod.aggregate_profile(raw)
        oracle_agg = ref.oracle_aggregate_profile(raw)
    except Exception:
        return m

    agg_match = True
    if set(learner_agg.keys()) != set(oracle_agg.keys()):
        agg_match = False
    else:
        for k in oracle_agg:
            for field in ["total_flops", "total_bytes", "total_time_us", "count", "intensity", "achieved_tflops"]:
                if abs(learner_agg[k][field] - oracle_agg[k][field]) > 1e-4:
                    agg_match = False
                    break
    if agg_match:
        m["aggregation_ok"] = 1.0

    try:
        learner_tot = intensity_mod.model_total_stats(learner_agg)
        oracle_tot = ref.oracle_model_total_stats(oracle_agg)
    except Exception:
        return m

    tot_match = True
    for field in ["total_flops", "total_bytes", "total_time_us", "overall_intensity", "overall_achieved_tflops"]:
        if abs(learner_tot[field] - oracle_tot[field]) > 1e-4:
            tot_match = False
            break
    if tot_match:
        m["totals_ok"] = 1.0

    return m
