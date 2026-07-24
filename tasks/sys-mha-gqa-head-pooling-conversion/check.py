import numpy as np

from mlsys import scorers


def _oracle_pool(K, V, n_kv_heads):
    B, H, T, D = K.shape
    r = H // n_kv_heads
    K_ref = K.reshape(B, n_kv_heads, r, T, D).mean(axis=2)
    V_ref = V.reshape(B, n_kv_heads, r, T, D).mean(axis=2)
    return K_ref, V_ref


def _per_head_channels(K, V):
    # (B, G, T, D) -> (G, B*T*D*2) so each GQA head is one "channel" row.
    B, G, T, D = K.shape
    Kt = K.transpose(1, 0, 2, 3).reshape(G, -1)
    Vt = V.transpose(1, 0, 2, 3).reshape(G, -1)
    return np.concatenate([Kt, Vt], axis=1)


def _cases():
    rng = np.random.default_rng(0)
    specs = [
        (1, 4, 2, 3, 2),
        (2, 8, 5, 4, 4),
        (2, 8, 5, 4, 2),
        (3, 6, 3, 8, 3),
        (1, 12, 4, 4, 1),  # degenerate MQA case
    ]
    out = []
    for B, H, T, D, G in specs:
        K = rng.standard_normal((B, H, T, D))
        V = rng.standard_normal((B, H, T, D))
        out.append((K, V, G))
    return out


def grade(sol, fx) -> dict:
    worst = 0.0
    for K, V, G in _cases():
        ref_K, ref_V = _oracle_pool(K, V, G)

        try:
            got_K, got_V = sol.mha_to_gqa_pool(K.copy(), V.copy(), G)
            got_K = np.asarray(got_K, dtype=np.float64)
            got_V = np.asarray(got_V, dtype=np.float64)
        except Exception:
            return {"channel_rel_err": float("inf")}

        if got_K.shape != ref_K.shape or got_V.shape != ref_V.shape:
            return {"channel_rel_err": float("inf")}
        if not (np.all(np.isfinite(got_K)) and np.all(np.isfinite(got_V))):
            return {"channel_rel_err": float("inf")}

        ref_channels = _per_head_channels(ref_K, ref_V)
        got_channels = _per_head_channels(got_K, got_V)

        err = scorers.channel_rel_err(ref_channels, got_channels, axis=1)
        worst = max(worst, err)

    return {"channel_rel_err": float(worst)}
