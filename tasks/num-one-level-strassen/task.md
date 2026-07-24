## Context

Matrix multiplication of two square matrices $A, B \in \mathbb{R}^{n \times n}$ produces

$$
C = AB,
$$

where each element is

$$
C_{ij} = \sum_{k=1}^{n} A_{ik}B_{kj}.
$$

Strassen's algorithm reduces the number of block multiplications by rearranging the computation. For one level, split matrices into four blocks:

$$
A =
\begin{bmatrix}
A_{11} & A_{12} \\
A_{21} & A_{22}
\end{bmatrix},
\qquad
B =
\begin{bmatrix}
B_{11} & B_{12} \\
B_{21} & B_{22}
\end{bmatrix}.
$$

Instead of eight block multiplications, compute seven products:

$$
\begin{aligned}
M_1 &= (A_{11}+A_{22})(B_{11}+B_{22}) \\
M_2 &= (A_{21}+A_{22})B_{11} \\
M_3 &= A_{11}(B_{12}-B_{22}) \\
M_4 &= A_{22}(B_{21}-B_{11}) \\
M_5 &= (A_{11}+A_{12})B_{22} \\
M_6 &= (A_{21}-A_{11})(B_{11}+B_{12}) \\
M_7 &= (A_{12}-A_{22})(B_{21}+B_{22})
\end{aligned}
$$

The output blocks are reconstructed as

$$
\begin{aligned}
C_{11} &= M_1 + M_4 - M_5 + M_7 \\
C_{12} &= M_3 + M_5 \\
C_{21} &= M_2 + M_4 \\
C_{22} &= M_1 - M_2 + M_3 + M_6.
\end{aligned}
$$

This task uses one Strassen level only. The seven products are ordinary block matrix multiplications. No recursive splitting is required.

## Task

Implement `one_level_strassen(A, B)`:

```python
def one_level_strassen(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    ...
```

The function receives two square NumPy arrays of the same shape. The dimension is guaranteed to be even. Return the matrix product using exactly the one-level Strassen block scheme.

Do not compute the full product directly with `A @ B` or `np.matmul(A, B)`. The seven block products are the intended multiplication operations.

## Example

```python
import numpy as np

A = np.array([[1., 2.],
              [3., 4.]])
B = np.array([[5., 6.],
              [7., 8.]])

C = one_level_strassen(A, B)

# C is approximately:
# [[19., 22.],
#  [43., 50.]]
```

## What the gate checks

The gate computes the reference result with NumPy's regular matrix multiplication oracle and compares the candidate output using relative L2 error:

$$
\mathrm{rel\_err} =
\frac{\lVert C_{\mathrm{candidate}} - C_{\mathrm{ref}} \rVert}
{\lVert C_{\mathrm{ref}} \rVert + 10^{-12}}.
$$

It also traces matrix multiplication operations performed by the candidate. The implementation must perform exactly seven block multiplications, matching the one-level Strassen construction. A direct full-matrix multiplication can produce the correct numbers but does not satisfy the algorithmic requirement.
