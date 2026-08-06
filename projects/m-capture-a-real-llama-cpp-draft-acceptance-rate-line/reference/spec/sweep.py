def find_optimal_draft_n(eval_func, n_values):
    best_n = None
    best_throughput = -1.0
    results = {}
    for n in n_values:
        throughput = eval_func(n)
        results[n] = throughput
        if throughput > best_throughput:
            best_throughput = throughput
            best_n = n
    return {"optimal_n": best_n, "optimal_throughput": best_throughput, "results": results}
