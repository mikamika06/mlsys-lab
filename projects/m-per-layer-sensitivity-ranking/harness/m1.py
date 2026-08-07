import ref


def check(workdir):
    from quant.sensitivity import compute_sensitivities

    out = {"ranking_match": 0.0}
    ok = 0
    total = len(ref.CONFIGS)
    for cfg in ref.CONFIGS:
        want = ref.compute_sensitivities(cfg, ref.STATS)
        got = compute_sensitivities(cfg, ref.STATS)
        if len(got) == len(want) and all(abs(a - b) < 1e-5 for a, b in zip(got, want)):
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"got {got}, reference {want}"
    if ok == total:
        out["ranking_match"] = 1.0
    return out
