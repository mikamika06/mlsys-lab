## Context

The natural logarithm of $1+x$ is a common building block in numerical algorithms.  
For very small values of $x$, the standard function $\log(1+x)$ suffers from catastrophic cancellation because the argument to the logarithm is close to one.  A stable alternative is the *log1p* routine, which evaluates

$$\operatorname{log1p}(x)=\log(1+x)$$

using a series expansion or other techniques that avoid subtracting nearly equal numbers.

The Taylor series for $\log(1+x)$ around $x=0$ is

$$
\log(1+x)=x-\frac{x^2}{2}+\frac{x^3}{3}-\frac{x^4}{4}+\cdots .
$$

When $|x|\ll 1$, the first term already gives an excellent approximation, but to achieve machine‑precision accuracy one must use a more sophisticated method such as the built‑in NumPy routine `np.log1p`.

## Task

Implement a function

```python
def log1p_tiny(x: np.ndarray) -> np.ndarray:
    ...
```

that takes a 2‑D or 1‑D NumPy array of type `float64` containing *tiny* values (e.g. $|x|\le10^{-12}$) and returns an array of the same shape with the accurate value of $\log(1+x)$ for each element.

The implementation must be fully vectorised, use only NumPy operations, and produce results that are within a relative error of $10^{-14}$ compared to `np.log1p`.

## Example

```python
import numpy as np
x = np.array([0.0, 1e-12, -5e-13])
y = log1p_tiny(x)
print(y)          # [0.00000000e+00 9.99999999e-13 -4.99999998e-13]
```

## What the gate checks

The grader evaluates your implementation against NumPy’s `np.log1p` on a set of randomly generated tiny values and computes the global relative L2 error

$$
\mathrm{rel\_err} = \frac{\|y_{\text{cand}}-y_{\text{ref}}\|}{\|y_{\text{ref}}\|}.
$$

The solution must satisfy $\mathrm{rel\_err}\le10^{-14}$.
