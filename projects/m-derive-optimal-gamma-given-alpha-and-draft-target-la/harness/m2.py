import ref


def check(workdir):
    from speculative.sweep import sweep_gamma_throughput

    alphas = [0.8, 0.5]
    cs = [0.2, 0.4]
    matched = 0
    for a in alphas:
        for c in cs:
            want_gamma = ref.compute_optimal_gamma(a, c, 8)
            sweep_results = sweep_gamma_throughput(a, c, list(range(1, 9)))
            best_g = min(sweep_results, key=sweep_results.get)
            if best_g == want_gamma:
                matched += 1

    out = {"argmin_index": 1.0 if matched >= 4 else 0.0, "throughput_matched": 1.0}
    return out
