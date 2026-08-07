def check(workdir):
    import ref
    from mfu.calculator import compute_attention_flops

    m = {"attention_flops_ok": 0.0}
    cfg = ref.get_sample_config()
    seq_len = 512

    try:
        val = compute_attention_flops(cfg, seq_len, causal=True)
        num_heads = cfg["num_heads"]
        head_dim = cfg["hidden_size"] // num_heads
        ref_val = 2 * num_heads * seq_len * seq_len * head_dim * 0.5
        if abs(val - ref_val) < 1e-5:
            m["attention_flops_ok"] = 1.0
    except Exception:
        pass
    return m
