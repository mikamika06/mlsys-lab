import ref


def check(workdir):
    from gkd.toy import compute_toy_divergence
    from gkd.drift import measure_distribution_drift

    out = {"divergences_matched": 0.0}
    ok = 0
    for i, (p, q, mode, beta) in enumerate(ref.TEST_CASES_TOY):
        want = ref.compute_toy_divergence(p, q, mode=mode, beta=beta)
        got = compute_toy_divergence(p, q, mode=mode, beta=beta)
        if got is not None and abs(want - got) < 1e-4:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"toy div {i}: got {got}, reference {want}"
    for i, (on_p, off_p) in enumerate(ref.TEST_CASES_DRIFT):
        want = ref.measure_distribution_drift(on_p, off_p)
        got = measure_distribution_drift(on_p, off_p)
        if got is not None and abs(want["tv_distance"] - got["tv_distance"]) < 1e-4:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"drift {i}: got {got}, reference {want}"
    out["divergences_matched"] = float(ok)
    return out
