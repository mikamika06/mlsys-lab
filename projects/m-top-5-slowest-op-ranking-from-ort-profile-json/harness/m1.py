import ref


def check(workdir):
    from ortprof.ranking import rank_top_slowest
    events = ref.generate_profile_data(100)
    got = rank_top_slowest(events)
    want = ref.rank_top_slowest(events)
    match = 1.0 if got == want else 0.0
    out = {"ranking_match": match}
    if match == 0.0:
        out["_note"] = f"got {got}, want {want}"
    return out
