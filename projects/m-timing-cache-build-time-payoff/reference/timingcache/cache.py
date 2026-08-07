def compute_payoff(build_times_no_cache, build_times_with_cache):
    return [n / max(c, 1e-5) for n, c in zip(build_times_no_cache, build_times_with_cache)]
