def check(workdir):
    import sys
    import ref

    sys.path.insert(0, workdir)
    import roofline.model as model_mod

    m = {"min_time_accurate": 0.0, "efficiency_ok": 0.0, "headroom_correct": 0.0}

    hw = ref.get_hw_spec()
    raw = ref.generate_raw_profile(44)
    agg = ref.oracle_aggregate_profile(raw)

    min_time_pass = True
    eff_pass = True
    headroom_pass = True

    try:
        for name, stat in agg.items():
            lb = model_mod.kernel_performance_bound(stat, hw)
            ob = ref.oracle_kernel_performance_bound(stat, hw)

            if abs(lb["min_time_us"] - ob["min_time_us"]) > 1e-3:
                min_time_pass = False
            if abs(lb["efficiency"] - ob["efficiency"]) > 1e-4:
                eff_pass = False
            if abs(lb["headroom_speedup"] - ob["headroom_speedup"]) > 1e-4:
                headroom_pass = False

        m["min_time_accurate"] = 1.0 if min_time_pass else 0.0
        m["efficiency_ok"] = 1.0 if eff_pass else 0.0
        m["headroom_correct"] = 1.0 if headroom_pass else 0.0
    except Exception:
        pass

    return m
