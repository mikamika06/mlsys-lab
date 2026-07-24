import numpy as np


def compare_eviction_methods(K: np.ndarray, V: np.ndarray, Q_hist: np.ndarray, q_new: np.ndarray,
                              budget: int, recent_window: int, snap_window: int,
                              pool_size: int) -> dict:
    """Compare three KV-cache eviction policies (Knorm, H2O, SnapKV) on the
    same context by selecting each policy's kept-token set and measuring
    the resulting single-query attention output error against full (no
    eviction) attention, plus the pairwise overlap of the kept sets.

    K, V        : (n, d) cached keys/values.
    Q_hist      : (T, d) queries already issued while this context was in
                  the KV cache (used by H2O/SnapKV to score tokens).
    q_new       : (d,) a new query to attend with, after eviction.
    budget      : number of tokens each policy is allowed to keep.
    recent_window : H2O's always-kept trailing window size (<= budget).
    snap_window : SnapKV's observation window size (<= budget, <= len(Q_hist)).
    pool_size   : odd kernel size for SnapKV's average pooling.

    Returns a dict with keys:
      "knorm_error", "h2o_error", "snapkv_error"       -- float, max abs
          error of that policy's compressed-KV attention output vs the
          full-context attention output for q_new.
      "overlap_knorm_h2o", "overlap_knorm_snapkv",
      "overlap_h2o_snapkv"                              -- int, size of the
          intersection of the two policies' kept index sets.
    """
    raise NotImplementedError('your code here')
