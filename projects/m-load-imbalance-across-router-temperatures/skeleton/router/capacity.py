import numpy as np


def compute_optimal_capacity_factor(
    logits: np.ndarray,
    temperature: float,
    top_k: int,
    num_experts: int,
    max_drop_rate: float = 0.0
) -> float:
    """Compute the minimum capacity factor required to keep token drop rate below max_drop_rate."""
    raise NotImplementedError
