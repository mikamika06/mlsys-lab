## Context

A Triton softmax kernel often assigns one program instance to each row of a
matrix. The program loads a block of values, applies a mask for positions beyond
the row width, computes the row maximum, and then performs a numerically stable
normalization.

For a vector $x \in \mathbb{R}^n$, softmax is

$$
\mathrm{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}} .
$$

Direct exponentiation can overflow for large values. A stable implementation
subtracts the row maximum $m = \max_i x_i$ before exponentiation:

$$
\mathrm{softmax}(x_i) =
\frac{e^{x_i-m}}{\sum_{j=1}^{n} e^{x_j-m}} .
$$

An emulated Triton kernel in Python should preserve this block-oriented
behavior while using NumPy operations instead of GPU instructions.

## Task

Implement `softmax_kernel(X)`:

```python
def softmax_kernel(X: np.ndarray) -> np.ndarray:
    ...
```

The input is a 2-D NumPy array of shape $(n, d)$. Treat each row as an
independent program instance. Return a `float64` NumPy array of the same shape
where each row is the stable softmax of the corresponding input row.

The implementation should avoid computing large exponentials before numerical
stabilization.

## Example

```python
import numpy as np

X = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
Y = softmax_kernel(X)

# Approximately:
# [[0.09003057, 0.24472847, 0.66524096],
#  [0.33333333, 0.33333333, 0.33333333]]
```

## What the gate checks

The gate computes a NumPy reference implementation using the stable softmax
equation

$$
\frac{\exp(x-m)}{\sum_j \exp(x_j-m)} .
$$

The maximum absolute error is measured over several generated inputs,
including rows with very large magnitudes. Non-finite outputs such as `nan` or
`inf` are treated as failing results.

The reported metric $\mathrm{max\_abs\_err}$ must satisfy

$$
\mathrm{max\_abs\_err} < 10^{-6}.
$$
