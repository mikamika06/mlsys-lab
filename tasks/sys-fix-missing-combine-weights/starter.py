import numpy as np

def moe_combine(expert_outputs, gate_weights):
    """Combine expert outputs using gate weights.

    Parameters
    ----------
    expert_outputs : np.ndarray, shape (n_experts, d)
    gate_weights   : np.ndarray, shape (n_experts,)

    Returns
    -------
    np.ndarray, shape (d,)
    """
    # BUG: gate_weights is never used — returns unweighted average
    return np.mean(expert_outputs, axis=0)
