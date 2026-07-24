## Context

In transformer models, attention scores are computed as dot products between query and key vectors. To enforce causality in autoregressive generation, a *causal mask* is applied so that each position can attend only to itself and previous positions. The mask is usually an additive matrix with $-\infty$ for forbidden entries, which after the softmax turns those probabilities into zero.

The softmax over a vector $z \in \mathbb{R}^n$ is

$$
\operatorname{softmax}(z)_i = \frac{\exp(z_i)}{\sum_{j=1}^{n}\exp(z_j)} .
$$

When an entry of $z$ is set to $-\infty$, $\exp(-\infty)=0$ and the corresponding probability becomes exactly zero.

## Task

Implement `causal_masked_softmax(scores)`:

```python
def causal_masked_softmax(scores: np.ndarray) -> np.ndarray:
    ...
```

`scores` is a 2‑D NumPy array of shape $(L, L)$ containing raw attention logits. The function must return an array of the same shape where each row has been softmaxed after applying a lower‑triangular causal mask (including the diagonal). The output should be of type `float64`.

## Example

```python
import numpy as np
scores = np.array([[0, 1, 2],
                   [3, 4, 5],
                   [6, 7, 8]], dtype=np.float64)

masked = causal_masked_softmax(scores)
print(masked)
# [[0.09003057 0.24472847 0.66524096]
#  [1.         0.         0.        ]
#  [1.         0.         0.        ]]
```

The first row is softmaxed normally; the second and third rows have all future positions masked to $-\infty$ before the softmax, so only the diagonal entry survives.

## What the gate checks

The grader computes a reference implementation using NumPy’s broadcasting and `np.exp`. It then evaluates the maximum absolute error between your output and the reference. Your solution must achieve

$$
\max_{i,j} |\, \hat{y}_{ij} - y_{ij}\,| \le 10^{-6}.
$$

The gate also verifies that the returned array has dtype `float64` and the same shape as the input.
