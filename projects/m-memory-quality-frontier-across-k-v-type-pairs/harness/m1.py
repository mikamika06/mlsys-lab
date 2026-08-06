import ref


def check(workdir):
    from kvquant.frontier import compute_pareto_frontier

    out = {"frontier_matched": 0.0}
    ok = 0
    total = len(ref.SAMPLE_MODELS)
    for cfg in ref.SAMPLE_MODELS:
        want = ref.compute_pareto_frontier(cfg, 4096, ref.SAMPLE_CANDIDATES)
        got = compute_pareto_frontier(cfg, 4096, ref.SAMPLE_CANDIDATES)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"got {got}, want {want}"
    if ok == total:
        out["frontier_matched"] = 1.0
    return out
