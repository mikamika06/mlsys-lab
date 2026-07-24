## Context

For a single example with logits $z \in \mathbb{R}^C$ and target class $t$, the
cross-entropy loss is
$$
\ell(z, t) = -\log\!\left(\frac{\exp(z_t)}{\sum_{c=0}^{C-1}\exp(z_c)}\right)
           = \operatorname{LSE}(z) - z_t,
\qquad
\operatorname{LSE}(z) = \log\!\sum_{c=0}^{C-1}\exp(z_c).
$$
Real logits can have magnitude in the thousands (e.g. before a temperature
scale, or from an unnormalised scoring head). Computing $\exp(z_c)$ directly
overflows to `inf` for $z_c \gtrsim 710$ in float64, which turns the loss into
`nan`. The fix is the log-sum-exp trick: subtract the row max before
exponentiating,
$$
\operatorname{LSE}(z) = m + \log\!\sum_{c}\exp(z_c - m), \qquad m = \max_c z_c,
$$
which is mathematically identical but never overflows.

## Task

Implement `fused_cross_entropy`:

```python
def fused_cross_entropy(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    ...
```

* `logits` — 2-D array of shape $(N, C)$, unnormalised scores (can be very large
  in magnitude).
* `targets` — 1-D integer array of length $N$, the correct class index per row.

Return a 1-D `float64` array of length $N$ with the per-example loss
$\ell_i = \operatorname{LSE}(z_i) - z_{i, t_i}$, computed **fused**: never
materialise the softmax probabilities, and use the max-subtraction trick so the
result stays finite for arbitrarily large `logits`. Fully vectorised — no
explicit Python loop over rows or classes.

## Example

```python
import numpy as np
logits = np.array([[1.0, 2.0, 3.0], [100000.0, 0.0, 0.0]])
targets = np.array([2, 0])
fused_cross_entropy(logits, targets)
# -> [0.4076059..., 0.0]   (row 1's loss is ~0: the target logit dominates)
```

## What the gate checks

`rel_err` — the grader loads a fixed fixture of logits whose magnitude ranges up
to $10^5$ (large enough that `np.exp(logits)` alone overflows to `inf`/`nan`),
computes the reference per-row loss in float64 with the log-sum-exp trick, and
compares it to your output with the global relative-L2-norm scorer. Gate:
`< 1e-10`. A naive (non-max-subtracted) implementation will emit `nan` on the
extreme rows and fail the gate; a scalar-mean implementation will fail the
shape check.
