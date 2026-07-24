## Context

The log‑sum‑exp function is a numerically stable way to compute the logarithm of a sum of exponentials:

$$\operatorname{logsumexp}(x) = \log\!\bigl(\sum_i e^{x_i}\bigr).$$

Direct evaluation can overflow or underflow when elements of $x$ are large in magnitude.  A common trick is to subtract the maximum element before exponentiating:

$$\operatorname{logsumexp}(x)
= \max(x) + \log\!\bigl(\sum_i e^{\,x_i-\max(x)}\bigr).$$

This shift preserves the result while keeping all intermediate values in a safe range.

## Task

Implement `logsumexp` that accepts a NumPy array `x` and an optional integer `axis`.  The function should return an array of the same type as `x`, but with the dimension specified by `axis` collapsed.  Use only vectorised NumPy operations; no Python loops are allowed.  The output must be of dtype `float64`.

```python
def logsumexp(x: np.ndarray, axis: int | None = None) -> np.ndarray:
    ...
```

The function should handle:

* `axis=None` – collapse all elements into a single scalar.
* Any valid integer `axis`, including negative indices.

## Example

```python
import numpy as np

x1 = np.array([-1., 0., 1.])
print(logsumexp(x1))
# 1.3132616875182227

x2 = np.array([[1., 2.], [3., 4.]])
print(logsumexp(x2, axis=1))
# array([2.40760596, 4.40760596])
```

## What the gate checks

The grader computes a reference implementation using NumPy’s own stable formula and compares your output to it with the metric `max_abs_err`.  The maximum absolute difference over all elements must be at most $10^{-10}$.
