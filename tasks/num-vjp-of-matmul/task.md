## Context

The matrix product of two real matrices $A \in \mathbb{R}^{m\times k}$ and $B \in \mathbb{R}^{k\times n}$ is defined by
$$
Y = A\,B,
\qquad Y_{ij} = \sum_{\ell=1}^{k} A_{i\ell}\,B_{\ell j}.
$$

In automatic differentiation the *vector‑Jacobian product* (VJP) of a function $f$ at a point $x$ with respect to an upstream gradient $\bar{y}$ is
$$
\bar{x} = \frac{\partial f}{\partial x}^{\!\top}\,\bar{y}.
$$

For the matrix multiplication $Y=A\,B$, the VJP with respect to $A$ and $B$ can be derived by the chain rule:
$$
\bar{A}= \bar{Y}\,B^{\!\top}, \qquad
\bar{B}= A^{\!\top}\,\bar{Y}.
$$

These formulas are used in many deep learning frameworks to propagate gradients through linear layers.

## Task

Implement the function `vjp_matmul` that computes the VJP of a matrix multiplication:

```python
def vjp_matmul(A: np.ndarray, B: np.ndarray, dY: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ...
```

* `A` is an `(m, k)` NumPy array,
* `B` is a `(k, n)` NumPy array,
* `dY` is the upstream gradient of shape `(m, n)`.

The function must return two arrays:
* `dA`, the gradient with respect to `A`, of shape `(m, k)`;
* `dB`, the gradient with respect to `B`, of shape `(k, n)`.

All computations should use NumPy only; no explicit Python loops are required. The result must be of dtype `float64`.

## Example

```python
import numpy as np

A = np.array([[1., 2.],
              [3., 4.]])
B = np.array([[5., 6.],
              [7., 8.]])
dY = np.array([[0.1, -0.2],
                [0.3, 0.4]])

dA, dB = vjp_matmul(A, B, dY)

print(dA)
# [[ 0.5  0.9]
#  [ 1.7  2.9]]

print(dB)
# [[ 0.6  1.8]
#  [ 1.2  3.4]]
```

## What the gate checks

The grader computes a reference VJP using central finite differences and compares it to your implementation with the metric `max_abs_err`. The candidate passes if
$$
\max_{i,j} \bigl|\,\text{candidate}_{ij} - \text{reference}_{ij}\,\bigr|
\le 10^{-5}.
$$

The reference is computed on random test cases of varying shapes, so a correct implementation will always satisfy the gate.
