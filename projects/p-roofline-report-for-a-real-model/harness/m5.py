def check(workdir):
    import sys
    import ref

    sys.path.insert(0, workdir)
    import roofline.analysis as analysis_mod

    m = {"validation_logic_ok": 0.0, "error_margin_ok": 0.0, "matches_flag_correct": 0.0}

    hw = ref.get_hw_spec()
    init_raw, opt_params, opt_raw = ref.generate_optimized_profile_pair(46)
    agg = ref.oracle_aggregate_profile(init_raw)

    try:
        pred = analysis_mod.estimate_optimization_speedup(
            agg, hw,
            target_kernels=opt_params["target_kernels"],
            memory_reduction_factor=opt_params["memory_reduction_factor"],
            target_efficiency=opt_params["target_efficiency"]
        )

        val = analysis_mod.validate_prediction_against_profile(pred, opt_raw)
        oracle_val = ref.oracle_validate_prediction_against_profile(pred, opt_raw)

        if abs(val["actual_time_us"] - oracle_val["actual_time_us"]) < 1e-3:
            m["validation_logic_ok"] = 1.0

        if abs(val["relative_error"] - oracle_val["relative_error"]) < 1e-4:
            m["error_margin_ok"] = 1.0

        if val["matches"] == oracle_val["matches"]:
            m["matches_flag_correct"] = 1.0
    except Exception:
        pass

    return m
