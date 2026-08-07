import numpy as np
from specdec.tracker import AcceptanceTracker
from specdec.model import SpeculativeModel
from specdec.policy import AdaptivePolicy


def generate_synthetic_traffic(n=100, seed=42):
    rng = np.random.RandomState(seed)
    domains = ["code", "chat", "summarization"]
    traffic = []
    for i in range(n):
        dom = domains[rng.choice(len(domains))]
        batch_size = int(rng.choice([1, 2, 4, 8, 16, 32]))
        if dom == "code":
            sim_acc = int(rng.binomial(5, 0.8))
        elif dom == "chat":
            sim_acc = int(rng.binomial(5, 0.5))
        else:
            sim_acc = int(rng.binomial(5, 0.2))

        traffic.append({
            "domain": dom,
            "batch_size": batch_size,
            "max_gamma": 5,
            "sim_accepted": sim_acc,
            "base_step_time": 10.0
        })
    return traffic


def run_reference_pipeline(traffic):
    tr = AcceptanceTracker(window_size=50)
    mod = SpeculativeModel(target_step_cost=10.0, draft_step_cost=1.0, overhead_per_draft=0.1)
    pol = AdaptivePolicy(mod, tr, min_speedup=1.02)
    return pol.evaluate_p95_and_throughput(traffic)
