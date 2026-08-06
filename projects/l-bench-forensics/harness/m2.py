import ref


def check(workdir):
    from benchkit import stats

    out = {"median_match": 1.0, "iqr_match": 1.0, "separable_match": 1.0,
           "separable_uses_iqr": 0.0}
    rows = ref.raw()
    samples = [r["samples_ts"] for r in rows if r.get("samples_ts")]
    for s in samples:
        e = ref.expect_stats(s)
        if not ref.near(stats.median(s), e["median"], 1e-12):
            out["median_match"] = 0.0
        if not ref.near(stats.iqr(s), e["iqr"], 1e-12):
            out["iqr_match"] = 0.0
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            if stats.separable(samples[i], samples[j]) != ref.expect_separable(
                    samples[i], samples[j]):
                out["separable_match"] = 0.0
    # Overlapping spreads must not be called apart, however far the medians are.
    a = [10.0, 20.0, 30.0]
    b = [11.0, 21.0, 31.0]
    if stats.separable(a, b) == 0 and stats.separable([1.0, 1.1, 1.2],
                                                      [5.0, 5.1, 5.2]) == 1:
        out["separable_uses_iqr"] = 1.0
    return out
