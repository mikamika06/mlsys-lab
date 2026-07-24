def grade(sol, fx) -> dict:
    """
    Compute reference values using the same formulas as in the task description,
    then compare with the solution's output.  No hard‑coded expected numbers are used.
    """
    cases = [
        (64, 8, 32),
        (128, 4, 16),
        (256, 16, 64),
        (512, 8, 128),
        (1024, 32, 256)
    ]
    ok = 1.0
    for d_model, heads, seq_len in cases:
        try:
            got = sol.compare_mha_vs_single(d_model, heads, seq_len)
            if not isinstance(got, tuple) or len(got) != 2:
                ok = 0.0
                break
            (params_mha, params_single), (flops_mha, flops_single) = got
        except Exception:
            ok = 0.0
            break

        dk = d_model // heads
        ref_params_mha = 3 * d_model * dk + d_model ** 2
        ref_params_single = 4 * d_model ** 2
        ref_flops_mha = heads * 4 * seq_len ** 2 * dk + 2 * seq_len * d_model ** 2
        ref_flops_single = 4 * seq_len ** 2 * d_model + 2 * seq_len * d_model ** 2

        if (params_mha, params_single) != (ref_params_mha, ref_params_single):
            ok = 0.0
            break
        if (flops_mha, flops_single) != (ref_flops_mha, ref_flops_single):
            ok = 0.0
            break

    return {"exact_match": ok}
