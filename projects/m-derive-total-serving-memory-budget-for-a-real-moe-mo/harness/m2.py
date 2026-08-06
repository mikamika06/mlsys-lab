import ref


def check(workdir):
    from moebudget.flops import compute_crossover_context
    out = {"crossover_match": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.compute_crossover_len(cfg)
        got = compute_crossover_context(cfg)
        if abs(got - want) < 1e-3 * want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, want {want}"
    if ok == len(ref.CONFIGS):
        out["crossover_match"] = 1.0
    return out
