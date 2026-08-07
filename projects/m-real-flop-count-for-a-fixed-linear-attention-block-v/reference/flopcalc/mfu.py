def calculate_mfu(tokens_per_sec, param_count, peak_flops_per_sec):
    executed_flops_per_sec = tokens_per_sec * (6.0 * param_count)
    return executed_flops_per_sec / peak_flops_per_sec
