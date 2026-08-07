def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    import ref
    from fa_fix import dispatcher

    m = {"ratio_measured": 0.0}
    q_bad = ref.create_tensor((1, 8, 128, 64), dtype="float32", aligned=False)
    try:
        ratio = dispatcher.measure_speedup(q_bad, q_bad, q_bad)
        if isinstance(ratio, (int, float)) and ratio > 1.0:
            m["ratio_measured"] = 1.0
    except Exception:
        pass
    return m
