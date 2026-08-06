import numpy as np

def compute_acceptance_metrics(cfg):
    probs = np.array(cfg["acceptance_probs"], dtype=float)
    gamma = int(cfg["gamma"])
    expected_accepted = float(np.sum(np.cumprod(probs)))
    avg_acceptance_rate = float(np.mean(probs))
    return {
        "expected_accepted": expected_accepted,
        "avg_acceptance_rate": avg_acceptance_rate,
        "gamma": gamma
    }
