from servemetrics.engine import simulate_generation

def compute_tpot_overhead(config, seed=42):
    t_unconstrained = simulate_generation(config, guided=False, seed=seed)
    t_guided = simulate_generation(config, guided=True, seed=seed)
    mean_unconstrained = sum(t_unconstrained) / len(t_unconstrained)
    mean_guided = sum(t_guided) / len(t_guided)
    ratio = mean_guided / mean_unconstrained
    return {
        "mean_unconstrained_tpot": mean_unconstrained,
        "mean_guided_tpot": mean_guided,
        "latency_ratio": ratio
    }
