def check(workdir):
    import ref
    from mfu.calculator import compute_mfu

    m = {"time_alignment_ok": 0.0}
    cfg = ref.get_sample_config()
    total_flops = 1e15
    measured_time = 10.0
    peak_tflops = 100.0

    try:
        val = compute_mfu(cfg, measured_time, total_flops, peak_tflops)
        expected = 1.0
        if abs(val - expected) < 1e-5:
            m["time_alignment_ok"] = 1.0
    except Exception:
        pass
    return m
