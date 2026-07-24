## Context

The sigmoid function is defined as $\sigma(z) = \frac{1}{1+e^{-z}}$.  
For large positive or negative $z$, computing $e^{-z}$ directly can overflow or underflow.  
A numerically stable implementation branches on the sign of $z$ and uses the identity
$$
\sigma(z)=\begin{cases}
\displaystyle \frac{1}{1+\exp(-z)} & z \ge 0,\\[6pt]
\displaystyle \frac{\exp(z)}{1+\exp(z)} & z < 0.
\end{cases}
$$
In the first case $\exp(-z)$ is small; in the second case $\exp(z)$ is small.  
This avoids overflow and preserves accuracy.

## Task

Implement `stable_sigmoid`:

```python
def stable_sigmoid(z: np.ndarray) -> np.ndarray:
    ...
```

The function accepts a NumPy array of arbitrary shape, returns an array of the same shape containing the sigmoid values in float64. No explicit Python loops are allowed; use vectorized operations only.

## Example

```python
import numpy as np
z = np.array([-1000, -1, 0, 1, 1000], dtype=np.float64)
y = stable_sigmoid(z)
# array([0., 0.26894142, 0.5, 0.73105858, 1.])
```

## What the gate checks

The grader computes a reference implementation using NumPy and compares your output with `arena.scorers.max_abs_err`.  
The maximum absolute difference must not exceed $10^{-12}$ over inputs in the range $[-1000,\;1000]$.  
A correct implementation should also avoid overflow for large magnitudes.
