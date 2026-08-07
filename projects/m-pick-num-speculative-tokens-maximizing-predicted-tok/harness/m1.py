import ref

def check(workdir):
    from spec.policy import pick_optimal_tokens
    out = {"optimal_matched": 0.0}
    alpha = 0.7
    mismatches = 0
    for model in ref.MODELS:
        want = ref.get_optimal_k(model, alpha)
        got = pick_optimal_tokens(model, alpha)
        if got != want:
            mismatches += 1
            if "_note" not in out:
                out["_note"] = f"model {model['name']}: got {got}, want {want}"
    if mismatches == 0:
        out["optimal_matched"] = 1.0
    return out
