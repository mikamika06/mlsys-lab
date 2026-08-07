import numpy as np

def ring_attention_simulate(q_shards, k_shards, v_shards):
    C = len(q_shards)
    out_shards = []

    for i in range(C):
        q = q_shards[i]
        o_unnorm = np.zeros_like(q, dtype=float)
        m = np.full((q.shape[0], 1), -np.inf, dtype=float)
        l = np.zeros((q.shape[0], 1), dtype=float)

        for step in range(C):
            j = (i - step) % C
            if j > i:
                continue

            k = k_shards[j]
            v = v_shards[j]

            scores = q @ k.T
            if j == i:
                mask = np.triu(np.ones_like(scores, dtype=bool), k=1)
                scores[mask] = -np.inf

            m_curr = np.max(scores, axis=-1, keepdims=True)
            m_new = np.maximum(m, m_curr)

            exp_scores = np.exp(scores - m_new)
            exp_old = np.exp(m - m_new)

            l_new = l * exp_old + np.sum(exp_scores, axis=-1, keepdims=True)
            o_unnorm = o_unnorm * exp_old + exp_scores @ v

            m = m_new
            l = l_new

        out_shards.append(o_unnorm / l)

    return out_shards
