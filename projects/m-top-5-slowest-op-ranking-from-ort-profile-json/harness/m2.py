import ref


def check(workdir):
    from ortprof.overhead import compute_overhead
    events = ref.generate_profile_data(200)
    got = compute_overhead(events)
    want = ref.compute_overhead(events)
    diff = abs(got - want)
    match = 1.0 if diff < 1e-5 else 0.0
    out = {"overhead_match": match}
    if match == 0.0:
        out["_note"] = f"got {got}, want {want}"
    return out
