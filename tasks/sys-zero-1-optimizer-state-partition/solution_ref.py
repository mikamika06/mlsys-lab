import math

def zero_one_adam(params: list[float], grads: list[list[float]], num_ranks: int, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> list[float]:
    """ZeRO-1 Adam: partition optimizer states across num_ranks.

    Maintains separate momentum and variance shards per rank.  Each rank r
    updates only its contiguous block of parameters at every step, then the
    full parameter vector is implicitly reconstructed (all-gather).
    """
    params = list(params)
    N = len(params)
    T = len(grads)

    base = N // num_ranks
    remainder = N % num_ranks
    starts = [0]
    for r in range(num_ranks):
        starts.append(starts[-1] + base + (1 if r < remainder else 0))

    m_shards = [[0.0] * (starts[r + 1] - starts[r]) for r in range(num_ranks)]
    v_shards = [[0.0] * (starts[r + 1] - starts[r]) for r in range(num_ranks)]

    for t in range(1, T + 1):
        for r in range(num_ranks):
            s, e = starts[r], starts[r + 1]
            g = grads[t - 1][s:e]

            m_shard = m_shards[r]
            v_shard = v_shards[r]

            m_new = [0.0] * len(m_shard)
            v_new = [0.0] * len(v_shard)

            for i in range(len(m_shard)):
                m_new[i] = beta1 * m_shard[i] + (1.0 - beta1) * g[i]
                v_new[i] = beta2 * v_shard[i] + (1.0 - beta2) * g[i] * g[i]

            m_shards[r] = m_new
            v_shards[r] = v_new

            m_hat = [0.0] * len(m_new)
            v_hat = [0.0] * len(v_new)

            bias1 = 1.0 - beta1 ** t
            bias2 = 1.0 - beta2 ** t

            for i in range(len(m_new)):
                m_hat[i] = m_new[i] / bias1
                v_hat[i] = v_new[i] / bias2

            for i in range(e - s):
                idx = s + i
                params[idx] -= lr * m_hat[i] / (math.sqrt(v_hat[i]) + eps)

    return params
