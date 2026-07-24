## Context

FlashAttention's backward pass needs the attention probability matrix
$P = \mathrm{softmax}(S)$ (where $S = QK^\top/\sqrt{d}$) again, but it was
never stored — only the tiny per-row scalar $\mathrm{lse}_i = \log\sum_j e^{S_{ij}}$
(the "logsumexp") was kept from the forward pass, since storing all of $P$
would cost $O(n^2)$ memory instead of $O(n)$. The backward pass recomputes
$P$ on the fly from $Q$, $K$, and the stored $\mathrm{lse}$:

$$
P_{ij} = \exp\!\big(S_{ij} - \mathrm{lse}_i\big)
       = \exp\!\left(\frac{(QK^\top)_{ij}}{\sqrt d} - \mathrm{lse}_i\right).
$$

This is mathematically identical to a numerically-stable softmax (since
$\mathrm{lse}_i = m_i + \log\sum_j e^{S_{ij}-m_i}$ for any shift $m_i$,
in particular the row max), but it reuses the *already-computed* stable
statistic instead of taking a fresh row max — which is the whole point:
during backward, no extra $O(n)$-per-row reduction is needed, just one
subtraction and one `exp`.

## Task

Implement `recompute_probs_from_lse`:

```python
def recompute_probs_from_lse(Q: np.ndarray, K: np.ndarray, lse: np.ndarray) -> np.ndarray:
    ...
```

* `Q` — array of shape $(n, d)$, query rows.
* `K` — array of shape $(m, d)$, key rows.
* `lse` — array of shape $(n,)$, the row-wise logsumexp of $S = QK^\top/\sqrt d$
  (computed and stored during the forward pass, given to you here).

Return $P$, shape $(n, m)$, with $P_{ij} = \exp(S_{ij} - \mathrm{lse}_i)$.
Each row of $P$ must sum to (approximately) 1.

**Do not** recompute a fresh row max or row sum from $S$ — use `lse` exactly
as given. Some of the grader's cases use large-magnitude $Q$, $K$ on purpose,
so that $S_{ij}$ itself is large enough that `exp(S)` overflows to `inf`
before any normalization; only subtracting the already-stable `lse` first
keeps every intermediate value finite.

## Example

```python
import numpy as np
Q = np.random.default_rng(0).normal(size=(3, 4))
K = np.random.default_rng(1).normal(size=(5, 4))
d = Q.shape[1]
S = (Q @ K.T) / np.sqrt(d)
m = S.max(axis=1, keepdims=True)
lse = m[:, 0] + np.log(np.exp(S - m).sum(axis=1))   # how lse was produced

P = recompute_probs_from_lse(Q, K, lse)
assert np.allclose(P.sum(axis=1), 1.0)
```

## What the gate checks

A single gate, **max_abs_err**, compares your output against
$\exp(QK^\top/\sqrt d - \mathrm{lse}[:, \text{None}])$ computed directly by
the grader, across several `(Q, K)` cases including ones whose raw scores
exceed `709` in magnitude (the point past which `exp` alone overflows
`float64`). Max absolute error must be `< 1e-6`; any `inf`/`nan` in your
output fails the case outright.
