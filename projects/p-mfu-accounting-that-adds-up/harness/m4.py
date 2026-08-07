def check(workdir):
    import ref
    from mfu.calculator import compute_total_flops

    m = {"phase_split_ok": 0.0}
    cfg = ref.get_sample_config()
    prefill_len = 128
    decode_steps = 10

    try:
        val = compute_total_flops(cfg, prefill_len, decode_steps)
        ref_val = ref.reference_total_flops(cfg, prefill_len, decode_steps)
        if abs(val - ref_val) < 1e-5:
            m["phase_split_ok"] = 1.0
    except Exception:
        pass
    return m
