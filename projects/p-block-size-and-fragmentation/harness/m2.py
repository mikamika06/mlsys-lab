def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import blk.analysis as learner
    import ref

    m = {"table_overhead_ok": 0.0}
    lengths = [10, 50, 100, 200]
    for bs in [16, 32, 64]:
        for ptr in [4, 8]:
            try:
                got = learner.block_table_overhead(lengths, bs, ptr)
                expected = ref.block_table_overhead(lengths, bs, ptr)
                if got != expected:
                    return m
            except Exception:
                return m
    m["table_overhead_ok"] = 1.0
    return m
