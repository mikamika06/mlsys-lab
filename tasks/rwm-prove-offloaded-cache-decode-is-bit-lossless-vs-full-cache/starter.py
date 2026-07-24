import numpy as np


def offloaded_decode_attention(Q, K_new, V_new):
    """Multi-layer, multi-head causal decode attention over a KV cache that
    lives in a CPU-resident "offload store" and is gathered back per step.

    Parameters
    ----------
    Q, K_new, V_new : np.ndarray, shape (L, T, H, d)
        L layers, T decode steps, H heads, d head dim.

    Returns
    -------
    out : np.ndarray, shape (L, T, H, d)

    BUG: at each decode step this only gathers the pair that was JUST
    pushed to the offload store, not the full history accumulated so far --
    so every step attends only to itself instead of every prior offloaded
    token. Fix the gather.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K_new = np.asarray(K_new, dtype=np.float64)
    V_new = np.asarray(V_new, dtype=np.float64)
    L, T, H, d = Q.shape
    scale = 1.0 / np.sqrt(d)
    out = np.zeros_like(Q)

    for l in range(L):
        cpu_k_store = []
        cpu_v_store = []
        for t in range(T):
            cpu_k_store.append(K_new[l, t])
            cpu_v_store.append(V_new[l, t])

            # only gathers the latest entry -- drops the offloaded past
            k_hist = cpu_k_store[-1][None, ...]
            v_hist = cpu_v_store[-1][None, ...]

            for h in range(H):
                q = Q[l, t, h, :]
                k = k_hist[:, h, :]
                v = v_hist[:, h, :]
                scores = (k @ q) * scale
                scores = scores - np.max(scores)
                w = np.exp(scores)
                w = w / np.sum(w)
                out[l, t, h, :] = w @ v

    return out
