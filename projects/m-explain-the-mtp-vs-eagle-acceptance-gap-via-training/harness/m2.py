import ref


def check(workdir):
    from mtpgap.simulation import simulate_acceptance_rates, compute_trajectory_divergence
    import numpy as np
    out = {"acceptance_match": 0.0, "divergence_match": 0.0}

    cfg = ref.CONFIGS[0]
    rates = simulate_acceptance_rates(cfg["mtp_probs"], cfg["eagle_probs"], cfg["temperature"])
    if isinstance(rates, dict) and "mtp_rate" in rates and "eagle_rate" in rates:
        out["acceptance_match"] = 1.0

    s1 = np.zeros((4, 16))
    s2 = np.ones((4, 16))
    div = compute_trajectory_divergence(s1, s2)
    if isinstance(div, float) and div > 0.0:
        out["divergence_match"] = 1.0

    return out
