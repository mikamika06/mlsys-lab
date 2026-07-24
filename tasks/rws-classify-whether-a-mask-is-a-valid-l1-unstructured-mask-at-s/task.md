## Context

PyTorch's `torch.nn.utils.prune.L1Unstructured` produces a binary mask
by ranking every weight by $|w|$ and pruning (masking to `False`) the
smallest fraction `amount` of them — the classic "magnitude pruning"
rule. A mask $M$ is a *valid* L1-unstructured mask for a tensor $w$ at
sparsity `amount` only if $M$ is **exactly** that smallest-magnitude
keep/prune split: right count pruned, and the *correct set* of indices
(not just any mask with the right sparsity level).

### The reference rule

For $w \in \mathbb{R}^n$ and `amount` $\in [0, 1]$, PyTorch first
converts the fraction to a count using Python's built-in `round`
(round-half-to-even):
$$
k = \mathrm{round}(\text{amount} \cdot n).
$$
If $k = 0$, every entry is kept ($M_i = \mathrm{True}\ \forall i$).
Otherwise, sort indices by $|w_i|$ ascending and prune (`False`) the $k$
smallest; every other index is kept (`True`).

A candidate mask $M$ is valid iff it equals this true mask **exactly**,
element for element — a mask with the right *number* of `False` entries
but the wrong *indices* is invalid.

## Task

Implement:

```python
def is_valid_l1_mask(w: np.ndarray, mask: np.ndarray, amount: float) -> bool:
    ...
```

* `w` — 1-D array of weights.
* `mask` — 1-D boolean array, same length as `w` (candidate keep-mask).
* `amount` — float in `[0, 1]`, the target pruning fraction.

Return `True` iff `mask` exactly equals the L1-unstructured mask defined
above for `(w, amount)`.

## Example

```python
import numpy as np
w = np.array([0.1, -0.9, 0.05, 0.4])
# amount=0.5 -> k = round(0.5*4) = 2 smallest |w| pruned: indices 0, 2
mask_true  = np.array([False, True, False, True])
mask_wrong = np.array([True, True, False, False])  # right count, wrong indices
is_valid_l1_mask(w, mask_true, 0.5)   # True
is_valid_l1_mask(w, mask_wrong, 0.5)  # False
```

## What the gate checks

* **exact_match** — your `True`/`False` verdict must match a NumPy
  oracle that builds the true mask via the exact rule above (including
  the `round()`-based count and ascending-`|w|` tie order) and compares
  it to the candidate, over many random `(w, mask, amount)` cases —
  roughly half genuinely valid, half corrupted by flipping a few mask
  bits (fixed seed).
