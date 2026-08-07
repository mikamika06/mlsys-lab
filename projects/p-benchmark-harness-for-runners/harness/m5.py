def check(workdir):
    from runner.core import check_consistency

    m = {"consistency_ok": 0.0}
    try:
        intervals = [(28, 32), (29, 33), (28.5, 32.5)]
        if check_consistency(intervals) is True:
            m["consistency_ok"] = 1.0
    except Exception:
        pass
    return m
