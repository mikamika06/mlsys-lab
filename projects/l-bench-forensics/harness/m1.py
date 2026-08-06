import ref


def check(workdir):
    from benchkit import parse

    rows = parse.load_all(ref.files())
    out = {"row_count": 0.0, "kind_match": 0.0, "throughput_match": 0.0,
           "ms_per_token_match": 0.0, "reps_match": 0.0}
    want = ref.raw()
    if len(rows) != len(want):
        return out
    out["row_count"] = 1.0
    kinds = tp = mpt = reps = 1.0
    for got_row, w in zip(rows, want):
        d = parse.derive(got_row)
        e = ref.expect_derive(w)
        if d.get("kind") != e["kind"]:
            kinds = 0.0
        if not ref.near(d.get("tokens_per_second", -1), e["tokens_per_second"], 1e-9):
            tp = 0.0
        if not ref.near(d.get("ms_per_token", -1), e["ms_per_token"], 1e-9):
            mpt = 0.0
        if d.get("reps") != e["reps"]:
            reps = 0.0
    out.update(kind_match=kinds, throughput_match=tp,
               ms_per_token_match=mpt, reps_match=reps)
    return out
