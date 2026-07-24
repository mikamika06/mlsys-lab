## Context

Matrix multiplication of two real matrices $A \in \mathbb{R}^{m\times p}$ and $B \in \mathbb{R}^{p\times n}$ produces a matrix $C \in \mathbb{R}^{m\times n}$ whose entries are defined by the bilinear form

$$
C_{ij} = \sum_{k=1}^{p} A_{ik}\, B_{kj}.
$$

A straightforward implementation evaluates this sum explicitly for every pair $(i,j)$ using three nested loops. The algorithm runs in $O(mnp)$ time and uses only elementary arithmetic operations on Python floats.

## Task

Implement the function `matmul(A, B)` that takes two 2‑D NumPy arrays of type ``float64`` with compatible shapes and returns their matrix product computed by a triple loop. Do **not** use any high‑level NumPy routine such as ``np.dot``, ``np.einsum`` or the ``@`` operator; only basic indexing, addition and multiplication are allowed.

```python
def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    ...
```

The returned array must have shape `(A.shape[0], B.shape[1])` and dtype `float64`.

## Example

```python
import numpy as np
A = np.array([[1., 2.],
              [3., 4.]])
B = np.array([[5., 6.],
              [7., 8.]])
C = matmul(A, B)
print(C)
# [[19. 22.]
#  [43. 50.]]
```

## What the gate checks

Two independent quality gates are applied:

1. **Numerical accuracy** – The maximum absolute difference between your result and NumPy’s reference ``np.dot`` must satisfy  
   $\displaystyle \max_{i,j}\lvert C_{\text{your}}(i,j)-C_{\text{ref}}(i,j)\rvert \le 10^{-6}$.

2. **BLAS prohibition** – Your implementation must not invoke any of the following: ``np.dot``, ``np.einsum`` or the ``@`` operator. The grader inspects your source code and fails if any of these are present.

Both gates must pass for the solution to be accepted.
