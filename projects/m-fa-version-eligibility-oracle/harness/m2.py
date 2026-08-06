import ref

def check(workdir):
    from fa_oracle import oracle
    out = {"eligibility_matched": 0.0, "configs": float(len(ref.ENVIRONMENTS))}
    ok = 0
    for i, env in enumerate(ref.ENVIRONMENTS):
        stack = ref.detect_stack_from_env(env)
        want = ref.evaluate_eligibility(stack)
        got = oracle.check_eligibility(stack)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"env {i}: got {got}, want {want}"
    out["eligibility_matched"] = float(ok)
    return out
