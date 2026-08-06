import ref


def check(workdir):
    from triton_calc.occupancy import max_resident_blocks
    from triton_calc.budget import evaluate_budget

    out = {"occupancy_matched": 0.0, "budget_matched": 0.0}
    occ_ok = True
    for tc in ref.TEST_CASES:
        want = ref.compute_max_blocks(
            tc["regs_per_thread"], tc["threads_per_block"], tc["spec"]
        )
        try:
            got = max_resident_blocks(
                tc["regs_per_thread"], tc["threads_per_block"], tc["spec"]
            )
            if got != want:
                occ_ok = False
        except Exception:
            occ_ok = False
    if occ_ok:
        out["occupancy_matched"] = 1.0

    try:
        want_budget = ref.evaluate_budget(ref.TEST_CASES)
        got_budget = evaluate_budget(ref.TEST_CASES)
        if got_budget == want_budget:
            out["budget_matched"] = 1.0
    except Exception:
        pass

    return out
