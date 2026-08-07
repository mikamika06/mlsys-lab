import numpy as np
from moe_balance.aux_loss import compute_switch_aux_loss
from moe_balance.bias_sim import simulate_deepseek_v3_bias_updates


def compare_convergence_speed(logits_sequence, alpha=0.01, gamma=0.1, top_k=1):
    """
    Compares expert load imbalance (CV) over time for Switch aux loss vs bias updates.
    """
    num_batches = len(logits_sequence)
    T, N = logits_sequence[0].shape

    # 1. Bias-based updates
    bias_res = simulate_deepseek_v3_bias_updates(
        logits_sequence, gamma=gamma, top_k=top_k
    )
    bias_loads = bias_res["load_history"]
    bias_cv = []
    for load in bias_loads:
        mean_l = np.mean(load)
        std_l = np.std(load)
        cv = std_l / mean_l if mean_l > 0 else 0.0
        bias_cv.append(cv)

    # 2. Aux loss gradient step simulation
    aux_loads = []
    curr_logits = [l.copy() for l in logits_sequence]
    lr = 0.1

    for step in range(num_batches):
        logits = curr_logits[step]
        _, grad = compute_switch_aux_loss(logits, alpha=alpha)

        # Apply gradient update to simulate routing adjustment
        logits = logits - lr * grad
        top1_idx = np.argmax(logits, axis=-1)
        counts = np.bincount(top1_idx, minlength=N).astype(np.float64)
        aux_loads.append(counts / T)

    aux_cv = []
    for load in aux_loads:
        mean_l = np.mean(load)
        std_l = np.std(load)
        cv = std_l / mean_l if mean_l > 0 else 0.0
        aux_cv.append(cv)

    return {
        "aux_cv": np.array(aux_cv),
        "bias_cv": np.array(bias_cv),
        "bias_converged_faster": bool(np.mean(bias_cv[-5:]) < np.mean(aux_cv[-5:])),
    }
