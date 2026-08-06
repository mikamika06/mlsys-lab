import ref


def check(workdir):
    from nvdiag.analyzer import diagnose_range
    events, _ = ref.generate_scenario()
    got = diagnose_range(events)
    want = {"wrong_tid": 2, "correct_tid": 1, "range_name": "Forward"}
    out = {"range_diagnosed": 0.0}
    if got == want:
        out["range_diagnosed"] = 1.0
    else:
        out["_note"] = f"got {got}, want {want}"
    return out
