def sweep_gamma_throughput(alpha, c, gamma_range):
    results = {}
    for g in gamma_range:
        s = sum(alpha ** k for k in range(g + 1))
        cost = (1.0 + c * g) / s
        results[g] = cost
    return results
