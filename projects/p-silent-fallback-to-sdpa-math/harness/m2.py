def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    import ref
    from fa_fix import dispatcher

    m = {"reason_found": 0.0}
    q_bad = ref.create_tensor((1, 8, 128, 64), dtype="float32", aligned=False)
    try:
        reason = dispatcher.find_disqualification_reason(q_bad, q_bad, q_bad)
        ref_reason = ref.find_disqualification_reason(q_bad, q_bad, q_bad)
        if reason == ref_reason and reason != "none":
            m["reason_found"] = 1.0
    except Exception:
        pass
    return m
