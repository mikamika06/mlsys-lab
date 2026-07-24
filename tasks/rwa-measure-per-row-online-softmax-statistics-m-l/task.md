## Context

The softmax function is defined for a vector $x \in \mathbb{R}^d$ as

$$\operatorname{softmax}(x)_i = \frac{\exp(x_i)}{\sum_{j=1}^{d}\exp(x_j)}.$$

Directly computing the denominator can lead to numerical overflow or under‑flow when elements of $x$ are large in magnitude. A common stabilisation trick is to subtract the maximum element $m=\max_j x_j$ before exponentiation:

$$\operatorname{softmax}(x)_i = \frac{\exp(x_i-m)}{\sum_{j=1}^{d}\exp(x_j-m)}.$$

The denominator $\displaystyle l = \sum_{j=1}^{d}\exp(x_j-m)$ is often called the *log‑sum‑exp* normaliser. For a batch of vectors stored as rows in a matrix $S\in\mathbb{R}^{n\times d}$ we wish to compute, for each row $i$,

$$m_i = \max_{j} S_{ij}, \qquad l_i = \sum_{j}\exp(S_{ij}-m_i).$$

These two statistics are useful in many machine‑learning pipelines: $m_i$ gives the log‑probability of the most likely class, while $\log(l_i)$ is the log‑partition function.

## Task

Implement a function `online_softmax_stats` that takes a 2‑D NumPy array `S` and returns two 1‑D arrays `(m, l)` containing the per‑row maximum and normaliser respectively. The implementation must use only NumPy operations; no explicit Python loops over rows or columns are allowed.

```python
def online_softmax_stats(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ...
```

The returned arrays should be of type `float64` and have shape `(S.shape[0],)`.

## Example

```python
import numpy as np
S = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])
m, l = online_softmax_stats(S)
print(m)   # [3. 6.]
print(l)   # [exp(0)+exp(-1)+exp(-2), exp(0)+exp(-1)+exp(-2)]
```

## What the gate checks

Two metrics are evaluated:

* `rel_err`: the maximum relative L₂ error between your output and a NumPy reference, averaged over several random test cases. Must be ≤ $10^{-6}$.
* `shape_ok`: verifies that both returned arrays have the correct shape `(n,)`. Must equal 1.

A correctly vectorised implementation will satisfy both gates. A naive or numerically unstable version will fail one of them.
