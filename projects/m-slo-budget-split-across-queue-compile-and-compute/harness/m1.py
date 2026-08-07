import ref


def check(workdir):
    from slobudget.budget import compute_breakdown

    ok = 0
    out = {"components_matched": 0.0}
    for c in ref.CASES:
        want = ref.compute_breakdown(c["slo"], c["queue_depth"], c["compile"], c["per_token"], c["tokens"])
        got = compute_breakdown(c["slo"], c["queue_depth"], c["compile"], c["per_token"], c["tokens"])
        if got and all(abs(got.get(k, 0) - want[k]) < 1e-4 for k in want):
            ok += 1
    out["components_matched"] = float(ok)
    return out
