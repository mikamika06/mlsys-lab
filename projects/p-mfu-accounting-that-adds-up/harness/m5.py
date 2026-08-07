def check(workdir):
    import ref
    from mfu.calculator import MFUCalculator

    m = {"tolerance_ok": 0.0}
    cfg = ref.get_sample_config()
    calc = MFUCalculator(cfg)
    workload = {"prefill_len": 64, "decode_steps": 5, "measured_time": 0.05, "peak_tflops": 312.0}

    try:
        mfu = calc.evaluate(workload)
        if 0.0 < mfu < 2.0:
            m["tolerance_ok"] = 1.0
    except Exception:
        pass
    return m
