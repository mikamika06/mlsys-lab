## Context

The **log‑sum‑exp** function is a numerically stable way to compute the logarithm of a sum of exponentials:

$$\operatorname{LSE}(x) = \log\!\Bigl(\sum_i e^{x_i}\Bigr).$$

Directly evaluating $\sum_i e^{x_i}$ can overflow or underflow when $x$ contains large positive or negative values. A common trick is to subtract the maximum element before exponentiating:

$$\operatorname{LSE}(x) = m + \log\!\Bigl(\sum_i e^{\,x_i-m}\Bigr), \qquad
m=\max_i x_i.$$

The **log‑softmax** of a vector $x$ is simply the log‑probability distribution obtained by normalising with $\operatorname{LSE}(x)$:

$$\operatorname{logSoftmax}(x)_i = x_i - \operatorname{LSE}(x).$$

Both operations are fundamental in machine learning, for example when computing cross‑entropy loss or sampling from categorical distributions.

## Task

Implement two functions that operate on arbitrary NumPy arrays and return `float64` results:

```python
def logsumexp(x: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Return the log-sum-exp of `x` along `axis`. If `axis` is None,
    collapse all dimensions into a single scalar."""
```

```python
def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return the element‑wise log-softmax of `x` along `axis`.
    The output has the same shape as `x`."""
```

Both functions must be fully vectorised (no explicit Python loops) and use only NumPy operations.

## Example

```python
import numpy as np
from your_module import logsumexp, log_softmax

a = np.array([0.1, 2.3, -5.0])
print(logsumexp(a))
# 2.3000000000000004

b = np.array([[1, 2], [3, 4]])
print(log_softmax(b))
# [[-1.31326169 -0.31326169]
#  [-1.31326169 -0.31326169]]
```

## What the gate checks

The grader evaluates your implementation against a reference computed with NumPy’s own stable routine (implemented in `check.py`). It reports the maximum absolute error over several random test cases:

- The error must satisfy $\mathrm{max\_abs\_err} \le 10^{-7}$.
- All outputs must be of type `float64`.

If either condition fails, the gate is not satisfied. No timing or operation‑count metrics are used for this task.
