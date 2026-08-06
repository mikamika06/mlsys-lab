def check(workdir):
    from accum.scaling import compute_gradient_inflation_factor
    f1 = compute_gradient_inflation_factor(8, False)
    f2 = compute_gradient_inflation_factor(8, True)
    match = 1.0 if (f1 == 8.0 and f2 == 1.0) else 0.0
    return {"inflation_matched": match}
