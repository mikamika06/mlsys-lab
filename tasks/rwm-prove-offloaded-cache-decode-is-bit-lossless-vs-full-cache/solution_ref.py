import math
import numpy as np


def offloaded_decode_attention(Q, K_new, V_new):
    """Multi-layer, multi-head causal decode attention over a KV cache that
    lives in a CPU-resident "offload store" and is gathered back per step.

    Parameters
    ----------
    Q, K_new, V_new : np.ndarray, shape (L, T, H, d)
        L layers, T decode steps (K_new[l, t], V_new[l, t] is the new key/
        value pair produced at decode step t for layer l), H heads, d head
        dim.

    Returns
    -------
    out : np.ndarray, shape (L, T, H, d)
        Attention output for every layer and every decode step.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K_new = np.asarray(K_new, dtype=np.float64)
    V_new = np.asarray(V_new, dtype=np.float64)
    L, T, H, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    out = np.zeros_like(Q)

    for l in range(L):
        cpu_k_store = []
        cpu_v_store = []
        for t in range(T):
            cpu_k_store.append(K_new[l, t])
            cpu_v_store.append(V_new[l, t])

            k_hist = np.stack(cpu_k_store, axis=0)
            v_hist = np.stack(cpu_v_store, axis=0)

            for h in range(H):
                q = Q[l, t, h, :]
                k = k_hist[:, h, :]
                v = v_hist[:, h, :]

                num_steps = t + 1
                scores = [0.0] * num_steps
                for step_idx in range(num_steps):
                    dot_val = 0.0
                    for dim_idx in range(d):
                        dot_val += k[step_idx, dim_idx] * q[dim_idx]
                    scores[step_idx] = dot_val * scale

                max_score = scores[0]
                for step_idx in range(1, num_steps):
                    if scores[step_idx] > max_score:
                        max_score = scores[step_idx]

                w = [0.0] * num_steps
                sum_w = 0.0
                for step_idx in range(num_steps):
                    val = math.exp(scores[step_idx] - max_score)
                    w[step_idx] = val
                    sum_w += val

                for step_idx in range(num_steps):
                    w[step_idx] /= sum_w

                out_vec = [0.0] * d
                for dim_idx in range(d):
                    acc = 0.0
                    for step_idx in range(num_steps):
                        acc += w[step_idx] * v[step_idx, dim_idx]
                    out_vec[dim_idx] = acc

                for dim_idx in range(d):
                    out[l, t, h, dim_idx] = out_vec[dim_idx]

    return out
