def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import blk.analysis as learner
    import ref

    m = {"optimum_ok": 0.0}
    lengths = [15, 20, 45, 90, 130]
    bs_list = [8, 16, 32, 64]
    try:
        got = learner.find_optimal_block_size(lengths, bs_list, 8, 2)
        expected = ref.find_optimal_block_size(lengths, bs_list, 8, 2)
        if got == expected:
            m["optimum_ok"] = 1.0
    except Exception:
        pass
    return m
