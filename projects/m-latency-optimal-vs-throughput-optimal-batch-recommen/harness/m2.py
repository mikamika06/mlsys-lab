import ref


def check(workdir):
    from batchopt.recommender import recommend_batches

    out = {"recommendations_matched": 0.0}
    ok = 0
    for i, p in enumerate(ref.PROFILES):
        want = ref.recommend_batches(p)
        got = recommend_batches(p)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"profile {i} recommendation mismatch: got {got}, want {want}"
    if ok == len(ref.PROFILES):
        out["recommendations_matched"] = 1.0
    return out
