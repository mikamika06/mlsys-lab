## Context

A Mixture-of-Experts (MoE) layer has $E$ experts but routes each token to
only $k \ll E$ of them. A small router network produces one logit per
expert per token, $z \in \mathbb{R}^{N\times E}$; each token's top-$k$
experts (by logit value) are selected, and the contribution of each
selected expert is weighted by a softmax computed **only over the selected
logits**:
$$
g_i = \operatorname{softmax}\big(z_{i,\text{top-}k}\big),\qquad
\sum_{r=1}^{k} g_{i,r} = 1 .
$$
Experts that weren't selected get exactly zero weight — this is what makes
MoE routing sparse (compute is only spent on the $k$ chosen experts, not
renormalized over all $E$).

## Task

Implement `topk_gating`:

```python
def topk_gating(logits: np.ndarray, k: int):
    ...
```

- `logits` — `(N, E)` float64 router logits, one row per token.
- `k` — number of experts to route each token to.

For each token (row) `i`:

1. Select the `k` experts with the highest `logits[i]` values, **ordered by
   decreasing logit value**. Break ties by **lower expert index first**
   (i.e. if two logits are exactly equal, the one with the smaller column
   index is considered "higher-ranked").
2. Compute the gate weight for each of those `k` selected logits as the
   softmax over just those `k` values (not over all `E`).

Return `(indices, weights)`:

- `indices` — `(N, k)` integer array, `indices[i]` = the selected expert
  indices for token `i`, in the rank order from step 1.
- `weights` — `(N, k)` float array, `weights[i, r]` = the gate weight for
  `indices[i, r]`, matching the same order.

## Example

```python
import numpy as np
logits = np.array([[0.1, 2.0, -1.0, 0.5]])
idx, w = topk_gating(logits, k=2)
# Highest two logits are 2.0 (expert 1) and 0.5 (expert 3).
# idx == [[1, 3]]
# w   == softmax([2.0, 0.5])  -- shape (1, 2), sums to 1
```

## What the gate checks

* **exact_match** — your `indices` array must exactly match a reference
  top-k selection (same experts, same rank order, same tie-break rule) on
  every test case, including a case with a deliberate tie between two
  experts' logits.
* **rel_err** — your `weights` array's relative L2 error against the
  reference softmax-over-selected weights must be `<= 1e-9`.
