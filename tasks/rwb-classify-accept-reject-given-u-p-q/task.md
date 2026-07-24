## Context

In many rejection‑sampling schemes we are given a scalar threshold $u\in[0,1]$ and two probability vectors $p,q\in[0,\infty)^n$. For each token $i$ the acceptance rule is

$$
\text{accept}_i \;\Longleftrightarrow\; u \leq \min\!\left(\frac{p_i}{q_i},\,1\right).
$$

The ratio $\frac{p_i}{q_i}$ is interpreted as an importance weight; it is capped at $1$ so that the acceptance probability never exceeds one. If $q_i=0$ we treat the ratio as zero, which guarantees rejection for that token unless $u=0$.

## Task

Implement `classify_accept(u, p, q)`:

```python
def classify_accept(u: float, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    ...
```

The function receives a scalar threshold `u`, and two 1‑D NumPy arrays of equal length. It must return a boolean array of shape `(len(p),)` where each element is `True` if the corresponding token should be accepted according to the rule above, otherwise `False`. The implementation must use only vectorised NumPy operations; no explicit Python loops are allowed.

## Example

```python
import numpy as np
p = np.array([0.5, 1.2])
q = np.array([1.0, 0.8])
u = 0.6
accept = classify_accept(u, p, q)
print(accept)          # [False  True]
```

## What the gate checks

The grader computes a reference implementation using NumPy and compares your output element‑wise with `np.array_equal`. The test suite includes deterministic edge cases (zero denominators, ratios larger than one, thresholds outside $[0,1]$) as well as random samples. Your solution must match the reference exactly for all tests; otherwise the gate fails.
