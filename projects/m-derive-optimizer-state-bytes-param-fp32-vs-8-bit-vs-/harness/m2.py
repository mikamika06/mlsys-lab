import ref


def check(workdir):
    from optbudget import budget as opt_budget
    from optbudget import spill as opt_spill

    out = {"budget_matched": 0.0, "step_matched": 0.0}
    budget_ok = True
    step_ok = True

    for i, cfg in enumerate(ref.CONFIGS):
        want_budget = ref.derive_total_memory_budget(cfg)
        try:
            got_budget = opt_budget.derive_total_memory_budget(cfg)
        except Exception as e:
            budget_ok = False
            out["_note"] = f"config {i} budget raised error: {type(e).__name__}"
            break
        if abs(float(got_budget) - float(want_budget)) > 1.0:
            budget_ok = False
            out["_note"] = f"config {i} budget got {got_budget}, reference {want_budget}"
            break

    for i, cfg in enumerate(ref.CONFIGS):
        want_step = ref.derive_spill_trigger_step(cfg)
        try:
            got_step = opt_spill.derive_spill_trigger_step(cfg)
        except Exception as e:
            step_ok = False
            out["_note"] = f"config {i} spill step raised error: {type(e).__name__}"
            break
        if int(got_step) != int(want_step):
            step_ok = False
            out["_note"] = f"config {i} step got {got_step}, reference {want_step}"
            break

    out["budget_matched"] = 1.0 if budget_ok else 0.0
    out["step_matched"] = 1.0 if step_ok else 0.0
    return out
