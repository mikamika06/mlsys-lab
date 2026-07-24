import numpy as np


def mla_gqa_equal_budget_compare(Q: np.ndarray, K: np.ndarray, V: np.ndarray, group_size: int):
    """Compare GQA(group_size) vs rank-matched truncated MLA at equal cache budget.

    Q: (batch, seq_q, n_heads, head_dim)
    K, V: (batch, seq_k, n_heads, head_dim) -- one KV pair per head, as in MHA.
    group_size: int dividing n_heads; GQA groups this many adjacent heads
        per shared KV pair (mean-pooled).

    Steps:
      1. Compute the true full-MHA output from the original Q, K, V.
      2. Compute the GQA(group_size) output: mean-pool K/V within each
         group of `group_size` heads, broadcast back to n_heads, run
         attention with the original Q. GQA's per-token cache budget is
         2 * n_kv_heads * head_dim scalars, where n_kv_heads = n_heads // group_size.
      3. Compute a truncated MLA reconstruction at the SAME per-token
         budget: concatenate the flattened K and V (per token) into one
         (2 * n_heads * head_dim)-wide matrix, take its best rank-`rank`
         approximation via SVD (rank = 2 * n_kv_heads * head_dim, matching
         GQA's budget), split the reconstruction back into K/V, and run
         attention with the original Q.
      4. gqa_err / mla_err = max absolute elementwise difference of each
         reconstruction's output vs the true MHA output.

    Returns (gqa_err: float, mla_err: float, winner: str), where winner is
    "mla" if mla_err < gqa_err else "gqa".
    """
    raise NotImplementedError('your code here')
