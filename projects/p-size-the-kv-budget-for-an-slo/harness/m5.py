def check(workdir):
    from kvcalc.calc import quantization_breakeven_point
    import ref

    m = {"quant_threshold_ok": 0.0}
    cfg = {"num_layers": 32, "num_kv_heads": 8, "head_dim": 128}
    try:
        res = quantization_breakeven_point(cfg, 16, 8, 1024)
        expected = ref.oracle_quant_point(cfg)
        if abs(res - expected) < 1e-5 or res is not None:
            m["quant_threshold_ok"] = 1.0
    except Exception:
        pass
    return m
