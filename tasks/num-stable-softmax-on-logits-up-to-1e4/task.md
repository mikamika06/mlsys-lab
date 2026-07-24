## Context

The softmax function maps a vector of logits $z \in \mathbb{R}^d$ to a probability distribution over $d$ classes:

$$\operatorname{softmax}(z)_i = \frac{\exp(z_i)}{\sum_{j=1}^{d}\exp(z_j)}.$$

When the components of $z$ are large (e.g. on the order of $10^4$), computing $\exp(z_i)$ directly can overflow, producing `inf` or `nan`.  A standard trick is to subtract the maximum logit from every entry before exponentiating:

$$\operatorname{softmax}(z)_i = \frac{\exp(z_i - m)}{\sum_{j=1}^{d}\exp(z_j - m)}, \qquad
m = \max_k z_k.$$

Because $z_i-m \le 0$, the exponentials are bounded by $1$ and never overflow.  This is called a *numerically stable* softmax.

## Task

Implement `stable_softmax(logits)`:

```python
def stable_softmax(logits: np.ndarray) -> np.ndarray:
    ...
```

`logits` is a 2‑D NumPy array of shape $(n, d)$ containing arbitrary real numbers up to $10^4$ in magnitude.  
The function must return an array of the same shape and type `float64`, where each row contains the softmax probabilities for that row’s logits.

Only vectorised NumPy operations are allowed; no Python loops.

## Example

```python
import numpy as np
logits = np.array([[0, 1, 2],
                   [10000, 9999, 9998]])
soft = stable_softmax(logits)
print(soft)
# [[0.09003057 0.24472847 0.66524096]
#  [0.73105858 0.26894142 0.00000000]]
```

The second row demonstrates that the large logits are handled without overflow.

## What the gate checks

* **Numerical correctness** – The mean Kullback–Leibler divergence between your output and a reference implementation is computed with `arena.scorers.mean_kl`.  It must be $\le 10^{-9}$.
* **Shape & dtype** – Your function must return an array of shape `(n, d)` and type `float64`; otherwise the gate fails.

The grader generates random logits in $[-10^4,\,10^4]$ for several test cases.  A naive implementation that does not subtract the row maximum will produce `inf`/`nan`, leading to a large KL divergence and failing the gate.
