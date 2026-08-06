import ref


def check(workdir):
    from moebudget.budget import compute_serving_memory
    out = {"budget_match": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.compute_total_memory(cfg)
        got = compute_serving_memory(cfg)
        if abs(got - want) < 1e-3 * want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, want {want}"
    if ok == len(ref.CONFIGS):
        out["budget_match"] = 1.0
    return out
