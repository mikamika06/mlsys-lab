def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    import ref
    from fa_fix import dispatcher

    m = {"backend_identified": 0.0}
    q_good = ref.create_tensor((1, 8, 128, 64), dtype="float16", aligned=True)
    q_bad = ref.create_tensor((1, 8, 128, 64), dtype="float32", aligned=False)

    try:
        b1 = dispatcher.get_backend(q_good, q_good, q_good)
        b2 = dispatcher.get_backend(q_bad, q_bad, q_bad)
        if b1 == "flash_attention" and b2 == "math":
            m["backend_identified"] = 1.0
    except Exception:
        pass
    return m
