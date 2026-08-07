def check(workdir):
    import ref
    m = {"threshold_ok": 0.0}
    history = [0.0] * 25
    if ref.oracle_disable(history, threshold=0.1) is True:
        if ref.oracle_disable([1.0] * 25, threshold=0.1) is False:
            m["threshold_ok"] = 1.0
    return m
