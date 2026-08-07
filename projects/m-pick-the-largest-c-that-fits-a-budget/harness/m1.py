import sys


def check(workdir):
    sys.path.insert(0, workdir)
    import ref
    from kvquant import check_flash_attn_requirement, fit_context_budget

    out = {"budgets_matched": 0.0, "flash_attn_checks_passed": 0.0}

    budget_ok = True
    for cfg, budget, tk, tv, blk in ref.TEST_BUDGETS:
        want = ref.fit_context_budget(cfg, budget, tk, tv, blk)
        got = fit_context_budget(cfg, budget, tk, tv, blk)
        if got != want:
            budget_ok = False
            out["_note"] = f"fit_context_budget failed: want {want}, got {got}"
            break

    if budget_ok:
        out["budgets_matched"] = 1.0

    fa_ok = True
    for tk, tv, fa_input, expected in ref.FLASH_ATTN_CASES:
        got = check_flash_attn_requirement(tk, tv, fa_input)
        if got != expected:
            fa_ok = False
            out["_note"] = f"check_flash_attn_requirement failed for {tk}/{tv}/fa={fa_input}: want {expected}, got {got}"
            break

    if fa_ok:
        out["flash_attn_checks_passed"] = 1.0

    return out
