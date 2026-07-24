## Context

In transformer decoding, logits are filtered before sampling: only the top‑$k$
tokens by logit value are kept, and the rest are masked with a large negative
number (so their probability under softmax becomes nearly zero).

A common branchy implementation sorts or loops with an `if` statement per
element:

```python
mask = np.full_like(logits, -np.inf)
vals = np.partition(logits, -k)[-k:]
thresh = np.min(vals)
for i in range(len(logits)):
    if logits[i] >= thresh:
        mask[i] = logits[i]
```

This conditional causes unpredictable branches and hurts SIMD throughput.

Using purely arithmetic and comparison operations instead of Python `if`
statements yields a **branchless** variant: compute a threshold $\tau$ of the
top‑$k$ logits, then build

$$ m_i = \mathrm{where}(x_i \ge \tau,\, x_i,\, -\infty) . $$

NumPy or any vectorized equivalent can implement this efficiently.

## Task

Implement `branchless_topk_mask(logits, k)`:

```python
def branchless_topk_mask(logits: np.ndarray, k: int) -> np.ndarray:
    """Return a copy of `logits` where only the top‑k entries are kept,
    others set to -inf, **without any Python branching**."""
```

It must:
1. Work for any 1‑D NumPy array of float64 logits and integer `k` $(1 \le k \le n)$.
2. Produce exactly the same mask values as the branchy reference.
3. Use vectorized arithmetic and comparisons only — no Python `if` or loops.

## Example

```python
import numpy as np
x = np.array([1.2, 0.3, 4.5, 2.2])
print(branchless_topk_mask(x, 2))
# keeps top‑2 (4.5, 2.2)
# array([-inf, -inf,  4.5,  2.2])
```

## What the gate checks

Two gates:

* **argmax_agreement**: ensures that the top‑$k$ positions of your mask match
  those of the reference (no mis‑ranking).
* **byte_exact_fraction**: verifies your output is bit‑identical to the
  reference mask values.

The grader builds random float64 vectors on a deterministic cache simulator,
computes both the branchy and your branchless mask, and checks agreement. Only
perfectly consistent and branchless outputs pass.
