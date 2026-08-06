import ref

def check(workdir):
    from specadapt.tuning import estimate_alpha, adaptive_gamma
    from specadapt.simulation import simulate_stream

    alphas = ref.generate_drifting_stream(seed=999, steps=300)

    def strat(hist, g):
        ae = estimate_alpha(hist)
        return adaptive_gamma(ae, g)

    adapt_perf = simulate_stream(alphas, strat)
    fixed_perf = ref.ref_fixed_simulate(alphas)

    ratio = adapt_perf / max(1e-6, fixed_perf)
    return {"throughput_ratio": float(ratio)}
