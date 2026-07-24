## Context

The dot product of two vectors $\mathbf{a},\mathbf{b}\in \mathbb{R}^n$ is defined as
$$
\langle \mathbf{a},\mathbf{b}\rangle = \sum_{i=1}^{n} a_i\,b_i .
$$

It is the most basic linear‑algebra operation and appears in every machine‑learning algorithm.

## Task

Implement `dot_product(a,b)` that takes two 1‑D NumPy arrays of equal length and returns their dot product as a Python float (or NumPy scalar). The implementation must use an explicit Python loop over the elements; no NumPy vectorised operations such as `np.dot` or broadcasting are allowed.

```python
def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    ...
```

The function should work for any array shape `(n,)`, with $n \ge 1$. The result must be a scalar of type `float64`.

## Example

```python
import numpy as np
a = np.array([1., 2., 3.])
b = np.array([4., 5., 6.])
print(dot_product(a, b))
# 32.0
```

## What the gate checks

The grader computes a reference dot product with `np.dot` on several random test cases and measures the global relative L² error

$$
\mathrm{rel\_err} = \frac{\lVert \hat y - y\rVert}{\lVert y\rVert + 10^{-12}} .
$$

The candidate must satisfy $\mathrm{rel\_err}\le 1\times10^{-9}$.
