## Context

The softmax function maps a vector of real numbers $z \in \mathbb{R}^d$ to a probability distribution over the same dimension:
$$
\operatorname{softmax}(z)_i = \frac{\exp(z_i)}{\sum_{j=1}^{d}\exp(z_j)}.
$$

When any component of $z$ is large, $\exp(z_i)$ can overflow to $+\infty$, and if the denominator also overflows the division may produce a NaN. A common fix is to subtract the maximum entry before exponentiating:
$$
\operatorname{softmax}(z)_i = \frac{\exp(z_i - m)}{\sum_{j=1}^{d}\exp(z_j - m)},\qquad m=\max_k z_k.
$$

This subtraction does not change the result because it cancels out in numerator and denominator.

## Task

Implement `stable_softmax(logits)`:

```python
def stable_softmax(logits: np.ndarray) -> np.ndarray:
    ...
```

It takes a 2‑D NumPy array of shape $(n, d)$ containing logits and returns an $(n, d)$ array of probabilities. The implementation must be fully vectorised (no Python loops) and use only NumPy operations.

## Example

```python
import numpy as np
logits = np.array([[0., 1., 2.], [1000., 1000., 1000.]])
probs = stable_softmax(logits)
print(probs)
# [[0.09003057 0.24472847 0.66524096]
#  [0.33333333 0.33333333 0.33333333]]
```

## What the gate checks

The mean Kullback–Leibler divergence $\mathrm{mean\_kl}$ between the reference softmax (computed with a numerically stable algorithm) and the candidate’s output must satisfy $\mathrm{mean\_kl} \le 10^{-9}$. A naive implementation that does not subtract the maximum will produce NaNs for large logits, causing the gate to fail.
