import ref


def check(workdir):
    from ttft.savings import simulate_batch

    max_err = 0.0
    for args in ref.M2_CASES:
        w_base, w_cache = ref.simulate_batch(*args)
        g_base, g_cache = simulate_batch(*args)
        err_base = abs(w_base - g_base) / (abs(w_base) + 1e-9)
        err_cache = abs(w_cache - g_cache) / (abs(w_cache) + 1e-9)
        max_err = max(max_err, err_base, err_cache)

    return {"rel_err": max_err}
