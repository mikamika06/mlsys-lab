import numpy as np


def moe_all_to_all_dispatch(X: np.ndarray, router_logits: np.ndarray,
                             expert_weight: np.ndarray, num_devices: int):
    """Simulate expert-parallel MoE dispatch: route -> all-to-all -> expert
    compute -> all-to-all back.

    See task.md for the exact routing/placement rule.
    """
    raise NotImplementedError('your code here')
