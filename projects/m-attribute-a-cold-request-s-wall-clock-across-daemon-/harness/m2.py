import ref


def check(workdir):
    from coldpath import phase_breakdown, classify_request, unattributed_ms
    from coldpath.timeline import cumulative_ms

    out = {"breakdown_match": 0.0, "classify_match": 0.0,
           "unattributed_match": 0.0, "timeline_match": 0.0}

    breakdown_ok = True
    classify_ok = True
    unattributed_ok = True
    timeline_ok = True

    for i, case in enumerate(ref.CASES):
        events = case["events"]

        want_bd = ref.phase_breakdown(events)
        got_bd = phase_breakdown(events)
        if got_bd != want_bd:
            breakdown_ok = False
            if "_note_breakdown" not in out:
                out["_note_breakdown"] = f"case {i}: got {got_bd}, reference {want_bd}"

        want_cls = ref.classify_request(events)
        got_cls = classify_request(events)
        if got_cls != want_cls:
            classify_ok = False
            if "_note_classify" not in out:
                out["_note_classify"] = f"case {i}: got {got_cls}, reference {want_cls}"

        want_un = ref.unattributed_ms(events)
        got_un = unattributed_ms(events)
        if got_un != want_un:
            unattributed_ok = False
            if "_note_unattributed" not in out:
                out["_note_unattributed"] = f"case {i}: got {got_un}, reference {want_un}"

        checkpoints = ref.checkpoints_for(events)
        want_tl = ref.cumulative_ms(events, checkpoints)
        got_tl = cumulative_ms(events, checkpoints)
        if got_tl != want_tl:
            timeline_ok = False
            if "_note_timeline" not in out:
                out["_note_timeline"] = f"case {i}: got {got_tl}, reference {want_tl}"

    out["breakdown_match"] = 1.0 if breakdown_ok else 0.0
    out["classify_match"] = 1.0 if classify_ok else 0.0
    out["unattributed_match"] = 1.0 if unattributed_ok else 0.0
    out["timeline_match"] = 1.0 if timeline_ok else 0.0
    return out
