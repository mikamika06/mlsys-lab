import ref


def check(workdir):
    from specprof.trace import parse_trace_events

    out = {"splits_matched": 0.0, "trace_extracted": 0.0}

    ok = 0
    for i, run in enumerate(ref.MOCK_RUNS):
        tot_d = sum(run["draft_times"])
        tot_t = sum(run["target_times"])
        tot_v = sum(run["verify_times"])
        tot_o = sum(run["overhead_times"])
        tot_all = tot_d + tot_t + tot_v + tot_o

        events = [
            {"name": "draft", "cat": "draft", "dur": tot_d},
            {"name": "target", "cat": "target", "dur": tot_t},
            {"name": "verify", "cat": "verify", "dur": tot_v},
            {"name": "overhead", "cat": "overhead", "dur": tot_o},
        ]

        want = ref.parse_trace_events(events)
        got = parse_trace_events(events)

        matched = True
        for k in ["draft", "target", "verify", "overhead"]:
            if abs(got.get(k, 0.0) - want[k]) > 1e-4:
                matched = False
                if "_note" not in out:
                    out["_note"] = (
                        f"run {i} phase {k}: got {got.get(k)}, want {want[k]}"
                    )
                break
        if matched:
            ok += 1

    out["splits_matched"] = float(ok)

    flattened = [ev for batch in ref.MOCK_TRACE_EVENTS for ev in batch]
    want_tr = ref.parse_trace_events(flattened)
    got_tr = parse_trace_events(flattened)

    tr_ok = True
    for k in ["draft", "target", "verify", "overhead"]:
        if abs(got_tr.get(k, 0.0) - want_tr[k]) > 1e-4:
            tr_ok = False
            break

    if tr_ok:
        out["trace_extracted"] = 1.0

    return out
