## Context

The softmax function maps a vector of logits $z \in \mathbb{R}^n$ to a probability distribution over $n$ classes:

$$\operatorname{softmax}(z)_i = \frac{\exp(z_i)}{\sum_{j=1}^{n}\exp(z_j)}.$$

When the logits are scaled by a temperature parameter $T>0$, the distribution becomes

$$\operatorname{softmax}_T(z)_i = \frac{\exp(z_i/T)}{\sum_{j=1}^{n}\exp(z_j/T)}.$$

A small temperature sharpens the distribution, while a large temperature smooths it. For numerical stability we usually subtract the maximum of $z/T$ before exponentiating.

## Task

Implement `softmax_temperature(logits, T)`:

```python
def softmax_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    ...
```

`logits` is a one‑dimensional NumPy array of arbitrary length.  
`T` is a positive scalar temperature.  
The function must return a NumPy array of the same shape and dtype `float64`, containing the probability distribution produced by applying softmax to `logits/T` in a numerically stable way.

## Example

```python
import numpy as np
from rws_temperature_softened_distribution import softmax_temperature

logits = np.array([1.0, 2.0, 3.0])
T = 0.5
probs = softmax_temperature(logits, T)
print(probs)   # [0.0025 0.0227 0.9748]
```

## What the gate checks

The grader computes a reference distribution using NumPy’s stable implementation and compares it to your output with the relative error metric:

$$\mathrm{rel\_err} = \frac{\lVert \hat p - p_{\text{ref}}\rVert}{\lVert p_{\text{ref}}\rVert},$$

where $\hat p$ is your result.  The gate requires $\mathrm{rel\_err}\le 10^{-8}$ on a suite of random test cases, including extreme logits that would overflow an unshifted exponentiation.
