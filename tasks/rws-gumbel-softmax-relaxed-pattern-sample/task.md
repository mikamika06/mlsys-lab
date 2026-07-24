## Context

Learning a **discrete** sparsity pattern (e.g. which 2 of every 4
weights to keep for N:M structured sparsity — there are $\binom{4}{2}=6$
valid 2:4 patterns per group) is not differentiable: you can't backprop
through an `argmax`. The **Gumbel-softmax** trick fixes this by relaxing
the discrete choice into a continuous, differentiable distribution over
the candidates: add Gumbel noise to the logits, divide by a temperature
$\tau$, and softmax. As $\tau \to 0$ this relaxation approaches a
one-hot sample from the categorical distribution defined by the logits;
at higher $\tau$ it's a soft, differentiable blend used during training,
later hardened to a discrete pick at inference.

For **grading determinism** this task takes the Gumbel noise as a fixed
input array (as if it had already been sampled once and cached) rather
than generating it internally — so your output is checked bit-for-bit
against a fixed-noise oracle instead of against a random distribution.

### Formula

Given logits $\ell \in \mathbb{R}^{\dots \times K}$ over $K$ candidate
patterns, fixed noise $g$ of the same shape (each entry a draw from a
standard Gumbel(0,1) distribution), and temperature $\tau > 0$:
$$
y_i = \frac{\exp\!\big((\ell_i + g_i)/\tau\big)}{\sum_j \exp\!\big((\ell_j + g_j)/\tau\big)}
$$
computed along the last axis (each row's $K$ candidates sum to 1).

## Task

Implement:

```python
def gumbel_softmax_relaxed(logits: np.ndarray, g: np.ndarray, tau: float) -> np.ndarray:
    ...
```

* `logits` — array of shape `(..., K)`.
* `g` — array of the same shape, fixed Gumbel(0,1) noise (already
  sampled — do not generate your own).
* `tau` — positive scalar temperature.

Return the softmax above, applied along the last axis, computed in a
numerically stable way (subtract the row max before exponentiating).

## Example

```python
import numpy as np
logits = np.array([[1.0, 0.0, -1.0]])
g = np.array([[0.2, -0.3, 0.1]])
y = gumbel_softmax_relaxed(logits, g, tau=0.5)
# y = softmax(([1.2, -0.3, -0.9]) / 0.5), sums to 1 along axis -1
```

## What the gate checks

* **max_abs_err** — your output must match a NumPy oracle computing
  `softmax((logits + g) / tau, axis=-1)` (numerically stable) to within
  $10^{-6}$ absolute error, over several random `(logits, g, tau)` cases
  with `K = 6` candidates (fixed seed) and varying batch sizes.
