def check(workdir):
    from kvcalc.calc import predict_trace_concurrency
    import ref

    m = {"trace_match_ok": 0.0}
    cfg = {"num_layers": 16, "num_kv_heads": 4, "head_dim": 64, "dtype_bytes": 2}
    trace = [128, 256, 512, 1024]
    limit = 1024 * 1024 * 100
    try:
        res = predict_trace_concurrency(cfg, trace, limit)
        expected = ref.oracle_trace_prediction(cfg, trace, limit)
        if abs(res - expected) / max(1, expected) <= 0.10:
            m["trace_match_ok"] = 1.0
    except Exception:
        pass
    return m
