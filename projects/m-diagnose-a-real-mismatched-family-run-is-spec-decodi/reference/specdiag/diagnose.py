import numpy as np
from specdiag.metrics import compute_acceptance_metrics

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
