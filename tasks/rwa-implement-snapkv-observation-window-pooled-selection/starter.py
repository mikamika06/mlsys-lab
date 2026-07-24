import numpy as np


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
    raise NotImplementedError('your code here')
