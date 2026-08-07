def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    import ref
    from fa_fix import dispatcher

    m = {"alignment_fixed": 0.0}
    q_bad = ref.create_tensor((1, 8, 128, 64), dtype="float32", aligned=False)
    try:
        q_f, k_f, v_f, _ = dispatcher.fix_inputs(q_bad, q_bad, q_bad)
        b = dispatcher.get_backend(q_f, k_f, v_f)
        if b == "flash_attention":
            m["alignment_fixed"] = 1.0
    except Exception:
        pass
    return m
