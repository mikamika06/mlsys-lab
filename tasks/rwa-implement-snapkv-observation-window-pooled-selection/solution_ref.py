import numpy as np


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _attend(q: np.ndarray, K: np.ndarray, V: np.ndarray, d: int) -> np.ndarray:
    scores = (q @ K.T) / np.sqrt(d)
    weights = _softmax_rows(scores[None, :])[0]
    return weights @ V


def snapkv_pooled_selection(K: np.ndarray, V: np.ndarray, Q_obs: np.ndarray, Q_new: np.ndarray,
                             budget: int, pool_size: int) -> dict:
    """SnapKV KV-cache compression, applied independently per attention
    head (each head may keep a different subset of positions).

    K, V   : (H, n, d) cached keys/values, per head.
    Q_obs  : (H, w, d) the last w queries issued while this context was
             cached (the 'observation window'), per head.
    Q_new  : (H, d) a new query per head, attended AFTER compression.
    budget : positions kept per head (>= w = Q_obs.shape[1]).
    pool_size : odd int, average-pooling kernel width over the token axis.

    For each head h:
      1. raw_score[i] = sum over the w observation-window queries of the
         softmax attention weight head h puts on position i.
      2. pooled_score = average-pool raw_score with `pool_size`
         (mode="edge" padding, output length n).
      3. Always keep the last w positions (the observation window
         itself). Fill the remaining budget - w slots with the top
         pooled_score positions OUTSIDE the window (ties broken by
         smaller index, via a stable descending sort).
      4. kept_idx[h] = sorted union of the window and the top picks.

    Returns a dict:
      "kept_idx": list of H sorted 1-D int arrays, kept[h] has length
                  budget (or w if budget <= w).
      "output":   (H, d) attention output of Q_new against the
                  compressed (kept-only) K/V, per head.
    """
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    Q_obs = np.asarray(Q_obs, dtype=np.float64)
    Q_new = np.asarray(Q_new, dtype=np.float64)

    H, n, d = K.shape
    w = Q_obs.shape[1]
    pad = pool_size // 2
    kernel = np.ones(pool_size) / pool_size

    kept_idx = []
    outputs = np.zeros((H, d), dtype=np.float64)

    for h in range(H):
        attn = _softmax_rows((Q_obs[h] @ K[h].T) / np.sqrt(d))  # (w, n)
        raw_score = attn.sum(axis=0)  # (n,)

        padded = np.pad(raw_score, (pad, pad), mode="edge")
        pooled = np.convolve(padded, kernel, mode="valid")  # (n,)

        win = np.arange(n - w, n)
        k_extra = budget - w
        if k_extra <= 0:
            idx = np.sort(win[-budget:])
        else:
            mask = np.ones(n, dtype=bool)
            mask[win] = False
            cand = np.nonzero(mask)[0]
            top_extra = cand[np.argsort(-pooled[cand], kind="stable")[:k_extra]]
            idx = np.sort(np.concatenate([win, top_extra]))

        kept_idx.append(idx)
        outputs[h] = _attend(Q_new[h], K[h][idx], V[h][idx], d)

    return {"kept_idx": kept_idx, "output": outputs}
