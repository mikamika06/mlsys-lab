def check(workdir):
    from kvcalc.calc import peak_headroom_capacity
    import ref

    m = {"headroom_ok": 0.0}
    try:
        res = peak_headroom_capacity(100, 1.5)
        expected = ref.oracle_headroom(100, 1.5)
        if abs(res - expected) < 1e-5:
            m["headroom_ok"] = 1.0
    except Exception:
        pass
    return m
