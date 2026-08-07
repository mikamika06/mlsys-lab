def check(workdir):
    import ref
    from mfu.calculator import compute_layer_flops

    m = {"flops_ok": 0.0}
    cfg = ref.get_sample_config()
    seq_len = 256

    try:
        val = compute_layer_flops(cfg, seq_len)
        ref_val = ref.reference_layer_flops(cfg, seq_len)
        if abs(val - ref_val) < 1e-5:
            m["flops_ok"] = 1.0
    except Exception:
        pass
    return m
