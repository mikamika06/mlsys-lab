def check(workdir):
    import sys
    import ref

    sys.path.insert(0, workdir)
    import roofline.model as model_mod

    m = {"ceiling_ok": 0.0, "classification_ok": 0.0, "bounds_ok": 0.0}

    hw = ref.get_hw_spec()

    try:
        c1 = model_mod.roofline_ceiling(10.0, hw)
        c2 = model_mod.roofline_ceiling(500.0, hw)
        o1 = ref.oracle_roofline_ceiling(10.0, hw)
        o2 = ref.oracle_roofline_ceiling(500.0, hw)
        if abs(c1 - o1) < 1e-3 and abs(c2 - o2) < 1e-3:
            m["ceiling_ok"] = 1.0
    except Exception:
        return m

    try:
        class_ok = True
        if model_mod.classify_kernel(5.0, hw) != "memory_bound":
            class_ok = False
        if model_mod.classify_kernel(200.0, hw) != "compute_bound":
            class_ok = False
        if class_ok:
            m["classification_ok"] = 1.0
    except Exception:
        return m

    raw = ref.generate_raw_profile(43)
    agg = ref.oracle_aggregate_profile(raw)
    sample_stat = next(iter(agg.values()))

    try:
        learner_b = model_mod.kernel_performance_bound(sample_stat, hw)
        oracle_b = ref.oracle_kernel_performance_bound(sample_stat, hw)

        b_ok = True
        for k in ["ceiling_flops_sec", "bound_type", "min_time_us", "achieved_flops_sec", "efficiency", "headroom_speedup"]:
            v_l = learner_b[k]
            v_o = oracle_b[k]
            if isinstance(v_o, str):
                if v_l != v_o:
                    b_ok = False
            else:
                if abs(v_l - v_o) > 1e-4:
                    b_ok = False
        if b_ok:
            m["bounds_ok"] = 1.0
    except Exception:
        pass

    return m
