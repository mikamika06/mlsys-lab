import ref


def check(workdir):
    from spectrain.acceptance import compute_acceptance_rate

    out = {"acceptance_rate_match": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.simulate_acceptance_rate(cfg["tokens"], cfg["draft_logits"], cfg["target_logits"], cfg["gamma"])
        try:
            got = compute_acceptance_rate(cfg["tokens"], cfg["draft_logits"], cfg["target_logits"], cfg["gamma"])
        except Exception:
            got = -1.0
        if abs(got - want) < 1e-4:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    if ok == len(ref.CONFIGS):
        out["acceptance_rate_match"] = 1.0
    return out
