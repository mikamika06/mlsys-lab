def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import blk.analysis as learner
    import ref

    m = {"internal_frag_ok": 0.0}
    lengths = [12, 33, 45, 60]
    for bs in [8, 16, 32, 64]:
        try:
            got = learner.internal_fragmentation(lengths, bs)
            expected = ref.internal_fragmentation(lengths, bs)
            if got != expected:
                return m
        except Exception:
            return m
    m["internal_frag_ok"] = 1.0
    return m
