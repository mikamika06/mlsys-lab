def compare_mha_vs_single(d_model: int, heads: int, seq_len: int):
    """
    Correct implementation of the formulas described in the task.
    All calculations are performed with Python integers.
    """
    dk = d_model // heads

    # Parameter counts
    params_mha = 3 * d_model * dk + d_model ** 2
    params_single = 4 * d_model ** 2

    # FLOP counts
    flops_mha = heads * 4 * seq_len ** 2 * dk + 2 * seq_len * d_model ** 2
    flops_single = 4 * seq_len ** 2 * d_model + 2 * seq_len * d_model ** 2

    return ((params_mha, params_single), (flops_mha, flops_single))
