def derive_optimal_gamma(alpha, c, max_gamma=8):
    best_g = 1
    best_val = float("inf")
    for g in range(1, max_gamma + 1):
        s = sum(alpha ** k for k in range(g + 1))
        val = (1.0 + c * g) / s
        if val < best_val:
            best_val = val
            best_g = g
    return best_g
