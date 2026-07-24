## Context

The softmax function converts a vector of real-valued logits $x \in \mathbb{R}^d$ into a
probability distribution:

$$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{d} e^{x_j}}$$

A fundamental mathematical property is **shift-invariance**: for any scalar
$c \in \mathbb{R}$,

$$\text{softmax}(x) = \text{softmax}(x - c) .$$

Subtracting the same constant from every component multiplies both numerator and
denominator by $e^{-c}$, which cancels:

$$\frac{e^{x_i - c}}{\sum_{j} e^{x_j - c}}
  = \frac{e^{-c}\, e^{x_i}}{e^{-c}\, \sum_{j} e^{x_j}}
  = \frac{e^{x_i}}{\sum_{j} e^{x_j}} .$$

This identity is the foundation of the numerically stable softmax. Choosing
$c = \max(x)$ shifts the largest logit to zero, so every exponent is $\le 0$ and
no overflow occurs:

$$\text{softmax}(x)_i
  = \frac{e^{x_i - \max(x)}}{\sum_{j} e^{x_j - \max(x)}} .$$

Without this trick a naive implementation $e^{x_i} / \sum_j e^{x_j}$ produces
`inf / inf = nan` whenever any $x_i$ exceeds roughly $709$ (the `float64`
overflow threshold for `np.exp`).

## Task

Implement `softmax(x)`:

```python
def softmax(x: np.ndarray) -> np.ndarray:
    ...
```

The function takes a 1-D NumPy `float64` array of length $d$ and must return a
1-D `float64` array of the same length containing the softmax probabilities.
Your implementation must be **numerically stable** — it must produce correct
results even when input values are very large (e.g.\ $x_i > 10^5$).

Do **not** call `scipy.special.softmax` or any library softmax; write the
computation yourself using NumPy.

## Example

```python
import numpy as np
x = np.array([1.0, 2.0, 3.0])
s = softmax(x)
# s ≈ [0.09003057, 0.24472847, 0.66524096]
assert abs(s.sum() - 1.0) < 1e-12
```

## What the gate checks

Two gates:

1. **accuracy** — For each test vector $x$ (including values larger than $10^5$),
   the maximum absolute difference $\lVert \text{softmax}(x) - \text{softmax}_{\text{ref}}(x)
   \rVert_\infty$ against a NumPy oracle (computed with the stable
   $c = \max(x)$ trick) must be $\le 10^{-12}$.

2. **shift_invariance** — For each test vector $x$ and several shift values
   $c \in \{0,\; \pm 1,\; \pm 100,\; \pm 1000\}$, the maximum absolute
   difference $\lVert \text{softmax}(x) - \text{softmax}(x - c) \rVert_\infty$
   must be $\le 10^{-12}$, directly demonstrating the identity.

A naive implementation without the max-subtraction trick overflows on large
inputs and fails both gates.
