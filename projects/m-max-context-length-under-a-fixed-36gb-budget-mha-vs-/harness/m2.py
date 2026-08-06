import ref


def check(workdir):
    from kvcalc import budget

    out = {"budget_matched": 0.0, "scaling_valid": 0.0}
    cfg = ref.CONFIGS[0]
    want_max = ref.max_context_length(cfg, ref.BUDGET_BYTES, 1)
    got_max = budget.max_context_length(cfg, ref.BUDGET_BYTES, 1)

    if got_max == want_max:
        out["budget_matched"] = 1.0
    else:
        out["_note"] = f"max context length mismatch: got {got_max}, reference {want_max}"

    # Verify scaling: MLA should support more context than MHA under the same budget
    mha_cfg = ref.CONFIGS[0]
    mla_cfg = ref.CONFIGS[2]
    mha_max = budget.max_context_length(mha_cfg, ref.BUDGET_BYTES, 1)
    mla_max = budget.max_context_length(mla_cfg, ref.BUDGET_BYTES, 1)

    if mla_max > mha_max:
        out["scaling_valid"] = 1.0
    else:
        if "_note" not in out:
            out["_note"] = f"scaling invalid: mla max {mla_max} not greater than mha max {mha_max}"

    return out
