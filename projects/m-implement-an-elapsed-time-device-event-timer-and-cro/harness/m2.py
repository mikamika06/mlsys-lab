import ref


def check(workdir):
    from timer.regimes import classify_regimes

    out = {"regimes_matched": 0.0, "rel_err": 0.0}
    got = classify_regimes(ref.REGIMES)
    want = ref.rank_regimes(ref.REGIMES)

    match = 1 if len(got) == len(want) and all(g["regime_id"] == w["regime_id"] for g, w in zip(got, want)) else 0
    out["regimes_matched"] = float(match)
    out["rel_err"] = 0.0
    return out
