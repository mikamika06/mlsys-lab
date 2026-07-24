## Context

The softplus function is a smooth approximation of the rectified linear unit (ReLU). It is defined by

$$\operatorname{softplus}(x)=\log(1+e^{\,x})\,.$$

For large positive arguments $x$, the exponential term $e^x$ overflows in floating‑point arithmetic, producing $\infty$ and rendering the function unusable. A numerically stable variant avoids this overflow by rewriting the expression as

$$
\operatorname{softplus}(x)=
\begin{cases}
x + \log(1+e^{-x}) & x>0\\[4pt]
\log(1+e^{\,x})   & x\le 0
\end{cases}
=
\max(0,x)+\log\!\bigl(1+e^{-|x|}\bigr)\,.
$$

Both forms use only $\log_1p$ and $e^{-x}$ (or $e^{-|x|}$), which remain finite for all real inputs.

## Task

Implement the function `softplus(x)` that accepts a NumPy array of arbitrary shape and returns an array of the same shape containing the element‑wise softplus values. The implementation must:

* use only NumPy operations (no Python loops);
* produce results with dtype `float64`;
* be numerically stable for inputs as large as $10^3$ in magnitude.

```python
def softplus(x: np.ndarray) -> np.ndarray:
    ...
```

## Example

```python
import numpy as np
x = np.array([-1000, -1, 0, 1, 1000], dtype=np.float64)
y = softplus(x)
print(y)
# [   0.00000000e+00   3.16227766e-01   6.93147181e-01
#    1.31326169e+00   1.00000000e+03]
```

## What the gate checks

The grader computes a reference implementation using the stable formula above and compares it to your output with the metric `max_abs_err`. The candidate passes only if

$$\max_{i} |\, \text{softplus}_{\text{candidate}}(x_i)-\text{softplus}_{\text{ref}}(x_i)\,| \le 10^{-10}\,. $$

The test inputs include values up to $\pm500$, which would overflow a naive implementation.
