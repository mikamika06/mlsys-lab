import numpy as np


def generate_synthetic_logits(num_batches=20, tokens=128, experts=8, seed=42):
    rng = np.random.RandomState(seed)
    seq = []
    for _ in range(num_batches):
        logits = rng.normal(size=(tokens, experts))
        logits[:, 0] += 2.5
        seq.append(logits)
    return seq


def compute_switch_aux_loss(logits, alpha=0.01):
    T, N = logits.shape
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    P = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    P_mean = np.mean(P, axis=0)

    top1_indices = np.argmax(logits, axis=-1)
    f = np.zeros(N, dtype=np.float64)
    for idx in top1_indices:
        f[idx] += 1.0
    f /= T

    loss = alpha * N * np.sum(P_mean * f)

    factor = (alpha * N) / T
    dot_f_P = np.sum(P * f, axis=-1, keepdims=True)
    grad = factor * P * (f - dot_f_P)

    return float(loss), grad


def simulate_deepseek_v3_bias_updates(logits_batch_sequence, gamma=0.1, top_k=2):
    if len(logits_batch_sequence) == 0:
        return {"biases": np.array([]), "load_history": np.array([])}

    T, N = logits_batch_sequence[0].shape
    e_bias = np.zeros(N, dtype=np.float64)
    bias_history = []
    load_history = []

    target_count = (T * top_k) / N

    for logits in logits_batch_sequence:
        adjusted_logits = logits + e_bias
        topk_indices = np.argsort(adjusted_logits, axis=-1)[:, -top_k:]

        counts = np.zeros(N, dtype=np.float64)
        for row in topk_indices:
            for idx in row:
                counts[idx] += 1.0

        load_history.append(counts / T)
        error = counts - target_count
        e_bias -= gamma * np.sign(error)
        bias_history.append(e_bias.copy())

    return {
        "biases": np.array(bias_history),
        "load_history": np.array(load_history),
    }


def compare_convergence_speed(logits_sequence, alpha=0.01, gamma=0.1, top_k=1):
    num_batches = len(logits_sequence)
    T, N = logits_sequence[0].shape

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

    aux_loads = []
    curr_logits = [l.copy() for l in logits_sequence]
    lr = 0.1

    for step in range(num_batches):
        logits = curr_logits[step]
        _, grad = compute_switch_aux_loss(logits, alpha=alpha)
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
