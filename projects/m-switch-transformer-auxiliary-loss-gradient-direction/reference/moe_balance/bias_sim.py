import numpy as np


def simulate_deepseek_v3_bias_updates(logits_batch_sequence, gamma=0.1, top_k=2):
    """
    Simulates DeepSeek-V3 aux-loss-free expert bias updates over a sequence of batches.
    logits_batch_sequence: list of arrays, each of shape (T, N).
    """
    num_batches = len(logits_batch_sequence)
    if num_batches == 0:
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

        # Update biases based on error relative to target
        error = counts - target_count
        e_bias -= gamma * np.sign(error)
        bias_history.append(e_bias.copy())

    return {
        "biases": np.array(bias_history),
        "load_history": np.array(load_history),
    }
