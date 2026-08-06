import numpy as np

CONFIGS = [
    {"gamma": 4, "t_draft": 1.5, "t_target": 8.0, "acceptance_probs": [0.8, 0.6, 0.4, 0.2]},
    {"gamma": 5, "t_draft": 2.0, "t_target": 10.0, "acceptance_probs": [0.3, 0.2, 0.1, 0.05, 0.02]},
    {"gamma": 3, "t_draft": 1.0, "t_target": 6.0, "acceptance_probs": [0.9, 0.8, 0.7]}
]

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

def diagnose_run(cfg):
    m = compute_acceptance_metrics(cfg)
    gamma = m["gamma"]
    e_acc = m["expected_accepted"]
    t_draft = float(cfg["t_draft"])
    t_target = float(cfg["t_target"])
    baseline_time_per_token = t_target
    spec_time_per_token = (t_draft * gamma + t_target) / (1.0 + e_acc)
    speedup = baseline_time_per_token / spec_time_per_token
    net_helping = bool(speedup > 1.0)
    breakeven_prob = t_draft / t_target
    return {
        "expected_accepted": e_acc,
        "speedup": float(speedup),
        "net_helping": net_helping,
        "breakeven_prob": float(breakeven_prob)
    }
