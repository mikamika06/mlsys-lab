import math


def offloaded_decode_attention(Q: list[list[list[list[float]]]], K_new: list[list[list[list[float]]]], V_new: list[list[list[list[float]]]]) -> list[list[list[list[float]]]]:
    """Multi-layer, multi-head causal decode attention over a KV cache that
    lives in a CPU-resident "offload store" and is gathered back per step.

    Parameters
    ----------
    Q, K_new, V_new : list[list[list[list[float]]]], shape (L, T, H, d)
        L layers, T decode steps (K_new[l][t], V_new[l][t] is the new key/
        value pair produced at decode step t for layer l), H heads, d head
        dim.

    Returns
    -------
    out : list[list[list[list[float]]]], shape (L, T, H, d)
        Attention output for every layer and every decode step.
    """
    L = len(Q)
    T = len(Q[0])
    H = len(Q[0][0])
    d = len(Q[0][0][0])
    scale = 1.0 / math.sqrt(d)

    out = [[[[0.0 for _ in range(d)] for _ in range(H)] for _ in range(T)] for _ in range(L)]

    for l in range(L):
        cpu_k_store = []
        cpu_v_store = []
        for t in range(T):
            cpu_k_store.append(K_new[l][t])
            cpu_v_store.append(V_new[l][t])

            k_hist = cpu_k_store
            v_hist = cpu_v_store

            for h in range(H):
                q = Q[l][t][h]

                scores = []
                for step_idx in range(len(k_hist)):
                    k = k_hist[step_idx][h]
                    dot_val = sum(q_i * k_i for q_i, k_i in zip(q, k))
                    scores.append(dot_val * scale)

                max_score = max(scores)
                exp_scores = [math.exp(s - max_score) for s in scores]
                sum_exp = sum(exp_scores)
                w = [es / sum_exp for es in exp_scores]

                out_vec = [0.0 for _ in range(d)]
                for step_idx in range(len(v_hist)):
                    v = v_hist[step_idx][h]
                    weight = w[step_idx]
                    for i in range(d):
                        out_vec[i] += weight * v[i]

                out[l][t][h] = out_vec

    return out
