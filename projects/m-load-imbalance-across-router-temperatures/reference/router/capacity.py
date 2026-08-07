import numpy as np
from router.assignments import analyze_router_assignments


def compute_optimal_capacity_factor(
    logits: np.ndarray,
    temperature: float,
    top_k: int,
    num_experts: int,
    max_drop_rate: float = 0.0
) -> float:
    analysis = analyze_router_assignments(logits, temperature, top_k, num_experts)
    counts = analysis["expert_counts"]
    num_tokens = logits.shape[0]
    base_capacity = (num_tokens * top_k) / num_experts

    for cf_int in range(10, 501):
        cf = cf_int / 10.0
        cap = int(np.ceil(base_capacity * cf))
        dropped = sum(max(0, c - cap) for c in counts)
        drop_rate = dropped / (num_tokens * top_k)
        if drop_rate <= max_drop_rate + 1e-9:
            return float(cf)

    return 5.0
