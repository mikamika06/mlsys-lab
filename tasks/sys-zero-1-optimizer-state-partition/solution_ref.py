import numpy as np

def zero_one_adam(params, grads, num_ranks, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """ZeRO-1 Adam: partition optimizer states across num_ranks.

    Maintains separate momentum and variance shards per rank.  Each rank r
    updates only its contiguous block of parameters at every step, then the
    full parameter vector is implicitly reconstructed (all-gather).
    """
    params = np.asarray(params, dtype=np.float64).copy()
    grads = np.asarray(grads, dtype=np.float64)
    N = len(params)
    T = len(grads)

    # --- contiguous shard boundaries ---
    base = N // num_ranks
    remainder = N % num_ranks
    starts = [0]
    for r in range(num_ranks):
        starts.append(starts[-1] + base + (1 if r < remainder else 0))

    # --- per-rank optimizer states ---
    m_shards = [np.zeros(starts[r + 1] - starts[r], dtype=np.float64)
                for r in range(num_ranks)]
    v_shards = [np.zeros(starts[r + 1] - starts[r], dtype=np.float64)
                for r in range(num_ranks)]

    for t in range(1, T + 1):
        for r in range(num_ranks):
            s, e = starts[r], starts[r + 1]
            g = grads[t - 1][s:e]

            # first and second moment update (shard-local)
            m_shards[r] = beta1 * m_shards[r] + (1.0 - beta1) * g
            v_shards[r] = beta2 * v_shards[r] + (1.0 - beta2) * g * g

            # bias correction
            m_hat = m_shards[r] / (1.0 - beta1 ** t)
            v_hat = v_shards[r] / (1.0 - beta2 ** t)

            # parameter update on shard only
            params[s:e] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    return params
