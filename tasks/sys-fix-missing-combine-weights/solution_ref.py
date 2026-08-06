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
    n_experts = expert_outputs.shape[0]
    d = expert_outputs.shape[1]
    res = []
    for j in range(d):
        acc = 0.0
        for i in range(n_experts):
            acc += gate_weights[i] * expert_outputs[i, j]
        res.append(acc)
    return np.array(res, dtype=expert_outputs.dtype)
