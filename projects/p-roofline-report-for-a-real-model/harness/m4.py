def check(workdir):
    import sys
    import ref

    sys.path.insert(0, workdir)
    import roofline.analysis as analysis_mod

    m = {"fusion_speedup_ok": 0.0, "ideal_speedup_ok": 0.0, "amdahl_bound_valid": 0.0}

    hw = ref.get_hw_spec()
    raw = ref.generate_raw_profile(45)
    agg = ref.oracle_aggregate_profile(raw)

    mem_kernels = [k for k, v in agg.items() if ref.oracle_classify_kernel(v["intensity"], hw) == "memory_bound"]

    try:
        learner_fusion = analysis_mod.estimate_optimization_speedup(
            agg, hw, target_kernels=mem_kernels, memory_reduction_factor=0.3, target_efficiency=1.0
        )
        oracle_fusion = ref.oracle_estimate_optimization_speedup(
            agg, hw, target_kernels=mem_kernels, memory_reduction_factor=0.3, target_efficiency=1.0
        )
        if abs(learner_fusion["speedup"] - oracle_fusion["speedup"]) < 1e-4:
            m["fusion_speedup_ok"] = 1.0
    except Exception:
        return m

    try:
        learner_ideal = analysis_mod.estimate_optimization_speedup(
            agg, hw, target_kernels=None, memory_reduction_factor=0.0, target_efficiency=1.0
        )
        oracle_ideal = ref.oracle_estimate_optimization_speedup(
            agg, hw, target_kernels=None, memory_reduction_factor=0.0, target_efficiency=1.0
        )
        if abs(learner_ideal["speedup"] - oracle_ideal["speedup"]) < 1e-4:
            m["ideal_speedup_ok"] = 1.0
    except Exception:
        return m

    try:
        if learner_ideal["speedup"] >= learner_fusion["speedup"] and learner_fusion["speedup"] >= 1.0:
            m["amdahl_bound_valid"] = 1.0
    except Exception:
        pass

    return m
