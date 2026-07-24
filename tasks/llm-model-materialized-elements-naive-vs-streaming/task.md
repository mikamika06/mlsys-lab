## Context

The softmax function is fundamental in machine learning. For a vector $z \in \mathbb{R}^d$ its probability distribution is

$$\sigma(z)_i = \frac{\exp(z_i)}{\sum_{j=1}^{d}\exp(z_j)}.$$

Direct evaluation can overflow when any component of $z$ is large, because $\exp(700)$ already exceeds the largest finite double. A common remedy is to subtract the maximum entry before exponentiation:

$$\sigma(z)_i = \frac{\exp(z_i - m)}{\sum_{j=1}^{d}\exp(z_j - m)}, \qquad m=\max_k z_k.$$

This “stable” form keeps all intermediate values bounded. When $z$ is a 2‑D array of shape $(n,d)$ we apply the formula row‑wise.

## Task

Implement `softmax_streaming(logits)` that accepts a NumPy array of shape `(n, d)` and returns an array of the same shape containing the softmax probabilities for each row. The implementation must use only vectorized NumPy operations; no explicit Python loops are allowed. It should be numerically stable even when elements of `logits` are very large or very small.

```python
def softmax_streaming(logits: np.ndarray) -> np.ndarray:
    ...
```

The result must have dtype `float64`.

## Example

```python
import numpy as np
logits = np.array([[1.0, 2.0, 3.0],
                   [1000.0, 1000.0, 1000.0]])
softmax_streaming(logits)
# array([[0.09003057, 0.24472847, 0.66524096],
#        [0.33333333, 0.33333333, 0.33333333]])
```

## What the gate checks

The grader computes a reference softmax using the stable formula above and compares your output with it via the global relative L2 error

$$\mathrm{rel\_err} = \frac{\lVert \hat y - y\rVert}{\lVert y\rVert + 10^{-12}}.$$

Your solution must satisfy $\mathrm{rel\_err}\le 10^{-9}$ on a set of test cases that include very large positive and negative logits. A naïve implementation that does not subtract the maximum will produce `inf`/`nan` for such inputs and fail this gate.
