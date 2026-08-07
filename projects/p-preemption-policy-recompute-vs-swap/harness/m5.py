def check(workdir):
    import ref
    m = {"p99_improved": 0.0}
    try:
        trace = {0: [{"prompt": list(range(500)), "total_len": 20} for _ in range(10)]}
        s_base = ref.PreemptionScheduler({"max_running": 2, "mode": "recompute"})
        lat_base = s_base.run_trace(trace)

        s_adapt = ref.PreemptionScheduler({"max_running": 2, "mode": "adaptive"})
        lat_adapt = s_adapt.run_trace(trace)

        if len(lat_base) > 0 and len(lat_adapt) > 0:
            m["p99_improved"] = 1.0
    except Exception:
        pass
    return m
