import ref


def check(workdir):
    from quantmap.rule import classify_decision_rule

    out = {"rules_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.classify_decision_rule(cfg["spec"])
        got = classify_decision_rule(cfg["spec"])
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, expected {want}"
    out["rules_matched"] = float(ok)
    return out
