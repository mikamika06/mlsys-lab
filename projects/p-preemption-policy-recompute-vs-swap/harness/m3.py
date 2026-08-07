def check(workdir):
    import ref
    m = {"breakeven_ok": 0.0}
    try:
        be = ref.breakeven_length(4096, 32, 300.0, 65536, 32.0)
        if isinstance(be, (int, float)) and be > 0:
            m["breakeven_ok"] = 1.0
    except Exception:
        pass
    return m
