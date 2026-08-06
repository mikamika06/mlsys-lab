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
    scale = 1.0 / np.sqrt(d)
    out = np.zeros_like(Q)

    for l in range(L):
        # The offload store: plain Python lists standing in for a CPU-side
        # buffer, indexed by decode step. New (k, v) pairs are pushed here
        # as they arrive; nothing that has ever been pushed is dropped.
        cpu_k_store = []
        cpu_v_store = []
        for t in range(T):
            # step t's own key/value pair is produced and pushed to the
            # offload store first...
            cpu_k_store.append(K_new[l, t])   # shape (H, d)
            cpu_v_store.append(V_new[l, t])

            # ...then, to compute this step's attention output, the FULL
            # history 0..t is gathered back from the offload store.
            k_hist = np.stack(cpu_k_store, axis=0)   # (t+1, H, d)
            v_hist = np.stack(cpu_v_store, axis=0)   # (t+1, H, d)

            for h in range(H):
                q = Q[l, t, h, :]              # (d,)
                k = k_hist[:, h, :]             # (t+1, d)
                v = v_hist[:, h, :]             # (t+1, d)
                scores = (k @ q) * scale        # (t+1,)
                scores = scores - np.max(scores)
                w = np.exp(scores)
                w = w / np.sum(w)
                out[l, t, h, :] = w @ v

    return out
