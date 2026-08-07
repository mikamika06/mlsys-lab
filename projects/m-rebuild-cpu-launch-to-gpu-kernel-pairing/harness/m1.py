import ref

def check(workdir):
    from trace_parser.pairing import pair_events
    events = ref.generate_trace()
    want = ref.pair_events(events)
    try:
        got = pair_events(events)
    except Exception:
        return {"pairs_matched": 0.0}
    if got == want:
        return {"pairs_matched": 1.0}
    return {"pairs_matched": 0.0}
