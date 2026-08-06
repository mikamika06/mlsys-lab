def compute_gradient_inflation_factor(accumulation_steps, normalize_by_accum_steps):
    if normalize_by_accum_steps:
        return 1.0
    return float(accumulation_steps)
