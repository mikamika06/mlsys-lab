import ref


def check(workdir):
    from rundiag import num_predict_budget

    out = {"budget_matched": 0.0, "bounded_by_context": 0.0, "infinite_ge_fill": 0.0}

    all_match = True
    for i, case in enumerate(ref.PREDICT_CASES):
        want = ref.num_predict_budget(*case)
        got = num_predict_budget(*case)
        if got != want:
            all_match = False
            if "_note" not in out:
                out["_note"] = f"case {i}: num_predict_budget{case} got {got}, reference {want}"
    out["budget_matched"] = 1.0 if all_match else 0.0

    bounded = True
    for num_predict, prompt_tokens, context_size, hard_cap in ref.PREDICT_CASES:
        if num_predict >= 0 or num_predict == -2:
            remaining = max(context_size - prompt_tokens, 0)
            got = num_predict_budget(num_predict, prompt_tokens, context_size, hard_cap)
            if got > remaining:
                bounded = False
    out["bounded_by_context"] = 1.0 if bounded else 0.0

    ge = True
    for _, prompt_tokens, context_size, hard_cap in ref.PREDICT_CASES:
        if hard_cap >= context_size:
            b_inf = num_predict_budget(-1, prompt_tokens, context_size, hard_cap)
            b_fill = num_predict_budget(-2, prompt_tokens, context_size, hard_cap)
            if b_inf < b_fill:
                ge = False
    out["infinite_ge_fill"] = 1.0 if ge else 0.0

    return out
