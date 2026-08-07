import ref


def check(workdir):
    from router.penalty import compute_penalty

    out = {"penalties_matched": 0.0}
    ok = 0
    for sc in ref.SCENARIOS:
        want = ref.compute_penalty(sc["cached"], sc["requested"], sc.get("cost", 1.0))
        got = compute_penalty(sc["cached"], sc["requested"], sc.get("cost", 1.0))
        if abs(got - want) < 1e-5:
            ok += 1
    out["penalties_matched"] = float(ok)
    return out
