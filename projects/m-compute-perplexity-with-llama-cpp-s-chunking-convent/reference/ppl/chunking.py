import numpy as np


def compute_perplexity(model, tokens, chunk_size):
    """Compute perplexity following llama.cpp chunking convention."""
    N = len(tokens)
    if N < 2:
        return 0.0

    total_loss = 0.0
    total_targets = 0

    for start in range(0, N, chunk_size):
        chunk = tokens[start : start + chunk_size]
        logits = model(chunk)
        L = len(chunk)

        for j in range(L):
            global_idx = start + j
            target_idx = global_idx + 1
            if target_idx < N:
                target_token = tokens[target_idx]
                pos_logits = np.asarray(logits[j], dtype=np.float64)
                m = np.max(pos_logits)
                lse = m + np.log(np.sum(np.exp(pos_logits - m)))
                total_loss += lse - pos_logits[target_token]
                total_targets += 1

    if total_targets == 0:
        return 0.0

    mean_loss = total_loss / total_targets
    return float(np.exp(mean_loss))
