def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import blk.analysis as learner
    import ref

    m = {"threshold_ok": 0.0}
    lengths = [16, 16, 16]
    bs = 16
    try:
        got = learner.check_memory_threshold(lengths, bs, 0.05)
        expected = ref.check_memory_threshold(lengths, bs, 0.05)
        if got == expected:
            m["threshold_ok"] = 1.0
    except Exception:
        pass
    return m
