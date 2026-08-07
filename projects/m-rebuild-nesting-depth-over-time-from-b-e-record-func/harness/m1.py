import ref


def check(workdir):
    from profparse.parser import parse_events
    events, _, _ = ref.get_sample_data()
    want = ref.parse_events(events) if hasattr(ref, "parse_events") else [
        {"name": "gemm", "pid": 1, "tid": 1, "ts": 120, "dur": 60, "ph": "X"},
        {"name": "forward", "pid": 1, "tid": 1, "ts": 100, "dur": 150, "ph": "X"}
    ]
    got = parse_events(events)

    # Check matching by content
    match = len(got) == len(want)
    if match:
        for g, w in zip(sorted(got, key=lambda x: x["ts"]), sorted(want, key=lambda x: x["ts"])):
            if g.get("name") != w.get("name") or g.get("dur") != w.get("dur") or g.get("ph") != w.get("ph"):
                match = False
                break
    return {"events_parsed": 1.0 if match else 0.0}
