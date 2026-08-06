import ref


def check(workdir):
    from quantfix.decision import should_use_model_free_ptq
    out = {"decision_match": 0.0}
    ok = 0
    for i, case in enumerate(ref.DECISION_CASES):
        got = should_use_model_free_ptq(case["hessian"])
        want = ref.require_model_free_ptq(case["hessian"])
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {got}, want {want}"
    if ok == len(ref.DECISION_CASES):
        out["decision_match"] = 1.0
    return out
