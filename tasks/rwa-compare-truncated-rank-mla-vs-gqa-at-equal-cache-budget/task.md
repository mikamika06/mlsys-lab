## Context

Both GQA and MLA shrink the per-token KV cache of a transformer, but they
throw information away in very different ways:

- **GQA($g$)**: partition the $n_h$ query heads into groups of $g$ adjacent
  heads and share one KV pair per group — here, the group's shared $K$/$V$
  is the mean of its members' original $K$/$V$. The cache stores
  $n_{kv}=n_h/g$ full-width KV pairs, i.e.
  $$
  \text{budget}_{\text{GQA}} = 2\, n_{kv}\, d_k \quad \text{scalars/token}.
  $$
- **MLA** (multi-head latent attention): instead of averaging heads
  together, jointly compress the *concatenation* of every head's $K$ and
  $V$ for a token, $m_t \in \mathbb{R}^{2 n_h d_k}$, down to its best
  rank-$r$ approximation and cache only that latent. By the
  Eckart–Young theorem, the minimum-Frobenius-error rank-$r$
  approximation of a matrix $M$ (stacking $m_t$ over all tokens) is given
  by truncating its SVD $M = U\Sigma V^\top$ to the top $r$ singular
  values:
  $$
  M_r = U_{:, :r}\, \Sigma_{:r, :r}\, V_{:, :r}^\top .
  $$
  The cache then stores only the $r$-wide latent code per token, so
  $\text{budget}_{\text{MLA}} = r$ scalars/token.

Setting $r = \text{budget}_{\text{GQA}} = 2 n_{kv} d_k$ gives MLA and GQA
**equal per-token cache budget** at that $g$. Because SVD truncation is the
*provably optimal* low-rank approximation (in Frobenius norm) of the joint
K/V matrix, while GQA's head-averaging is a much cruder, structurally
constrained compressor, MLA often — but not always, it depends on the
actual structure of the data at that specific budget — reconstructs the
true full-MHA attention output more accurately at the same budget.

## Task

Implement `mla_gqa_equal_budget_compare(Q, K, V, group_size)`:

```python
def mla_gqa_equal_budget_compare(Q: np.ndarray, K: np.ndarray, V: np.ndarray, group_size: int):
    ...
```

- `Q`: shape `(batch, seq_q, n_heads, head_dim)`.
- `K`, `V`: shape `(batch, seq_k, n_heads, head_dim)` — one KV pair per
  head, as in ordinary MHA.
- `group_size`: a positive int dividing `n_heads`.

Steps:

1. Compute the true full-MHA output `mha_out` from the original `Q, K, V`
   (standard scaled dot-product attention, scale `1/sqrt(head_dim)`).
2. Compute the **GQA(`group_size`)** output: reshape the head axis of
   `K`/`V` into `(n_heads // group_size, group_size)` groups, mean-pool
   over the group axis, broadcast (repeat) each group's pooled KV back
   out to its `group_size` heads, then run attention with the original
   `Q`. Let `n_kv = n_heads // group_size`.
3. Compute the **MLA** output at the *same* per-token budget: flatten `K`
   and `V` per token to width `n_heads * head_dim` each, concatenate them
   into one `(batch, seq_k, 2 * n_heads * head_dim)` array, take its
   per-batch, per-token-axis truncated SVD to rank
   `rank = 2 * n_kv * head_dim` (i.e. `min(rank, num_singular_values)`
   if `rank` exceeds what's available), reconstruct
   `M_r = U_r @ diag(S_r) @ Vt_r`, split the last axis back into the
   reconstructed `K` and `V` halves (`n_heads * head_dim` wide each,
   `K` first), reshape back to `(batch, seq_k, n_heads, head_dim)`, and
   run attention with the original `Q`.
4. `gqa_err = max(|gqa_out - mha_out|)`, `mla_err = max(|mla_out - mha_out|)`
   (max over every element).
5. `winner = "mla"` if `mla_err < gqa_err`, else `winner = "gqa"`.

Return `(gqa_err, mla_err, winner)`.

Use `numpy.linalg.svd(..., full_matrices=False)`, which batches over
leading dimensions, so no explicit loop over `batch` is required.

## Example

```python
import numpy as np

rng = np.random.default_rng(0)
Q = rng.standard_normal((2, 48, 8, 4))
K = rng.standard_normal((2, 48, 8, 4))
V = rng.standard_normal((2, 48, 8, 4))

gqa_err, mla_err, winner = mla_gqa_equal_budget_compare(Q, K, V, group_size=2)
print(gqa_err, mla_err, winner)
# gqa_err ~ 1.19, mla_err ~ 0.58, winner = "mla"
```

## What the gate checks

The oracle builds several `(Q, K, V, group_size)` cases from a seeded
generator, spanning different batch sizes, sequence lengths, head counts,
and head dims, with `group_size` values ranging from mild GQA up through
full MQA (`group_size == n_heads`). For each case it independently computes
`gqa_out`/`mla_out` exactly as described (mean-pool-and-broadcast for GQA,
true SVD truncation via `np.linalg.svd` for MLA — this is the real oracle,
not a hardcoded winner), and derives `gqa_err`, `mla_err`, and `winner`.

Your `(gqa_err, mla_err)` values are compared to the oracle's
(`max_abs_err`, threshold `1e-4`) across all cases, and your `winner`
string is checked for exact equality against the oracle's winner in every
case (`exact_match`, must be `1.0`, i.e. every case's winner must match —
the test cases are constructed so the winner is **not** always `"mla"`,
so a solution that hardcodes `"mla"` will fail on at least one case).
Reversing the equal-budget rank formula, pooling GQA the wrong way, or
truncating the SVD to the wrong rank will throw off both the error values
and, often, the winner.
