def check(workdir):
    from kvcalc.calc import kv_bytes_per_token
    import ref

    m = {"formula_ok": 0.0}
    cfg = {"num_layers": 32, "num_kv_heads": 8, "head_dim": 128, "dtype_bytes": 2}
    try:
        res = kv_bytes_per_token(cfg)
        expected = ref.oracle_kv_bytes(cfg)
        if abs(res - expected) < 1e-5:
            m["formula_ok"] = 1.0
    except Exception:
        pass
    return m
