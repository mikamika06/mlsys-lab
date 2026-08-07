def check(workdir):
    from kvcalc.calc import effective_capacity
    import ref

    m = {"fragmentation_ok": 0.0}
    try:
        res = effective_capacity(1024 * 1024 * 1024, 16, 1000)
        expected = ref.oracle_fragmentation(1024 * 1024 * 1024, 16, 1000)
        if abs(res - expected) / max(1, expected) < 0.05:
            m["fragmentation_ok"] = 1.0
    except Exception:
        pass
    return m
