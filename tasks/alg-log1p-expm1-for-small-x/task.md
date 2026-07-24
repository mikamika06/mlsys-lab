## Context

The functions $\log(1+x)$ and $\exp(x)-1$ arise frequently in statistical models.  
For $x$ close to zero the naive implementations suffer from catastrophic cancellation: computing $1+x$ loses precision when $x\ll 1$, and then taking a logarithm or an exponential amplifies that loss. The NumPy primitives `np.log1p` and `np.expm1` are designed to avoid this by using specialised algorithms.

## Task

Implement two functions:

```python
def stable_log1p(x: np.ndarray | float) -> np.ndarray | float:
    ...
```

and

```python
def stable_expm1(x: np.ndarray | float) -> np.ndarray | float:
    ...
```

Both take a scalar or 1‑D NumPy array of real numbers and return the numerically stable value of $\log(1+x)$ and $\exp(x)-1$ respectively. The result must have type `float64` when an array is passed, and should preserve the input shape.

## Example

```python
import numpy as np
x = np.array([0.0, 1e-8, -1e-8])
print(stable_log1p(x))
# [0.00000000e+00 9.99999995e-09 -9.99999995e-09]
print(stable_expm1(x))
# [0.00000000e+00 1.00000001e-08 -1.00000001e-08]
```

## What the gate checks

Two gates evaluate the relative L2 error of your implementation against NumPy’s reference functions on a set of random values in $[-10^{-8},\,10^{-8}]$.  
The maximum relative error over all test cases must be at most $10^{-12}$ for both `stable_log1p` and `stable_expm1`.  
A naive use of `np.log(1+x)` or `np.exp(x)-1` will not satisfy this precision requirement.
