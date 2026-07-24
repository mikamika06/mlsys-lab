## Context

The Sylvester-Hadamard construction recursively builds square matrices whose entries are $1$ or $-1$.

Starting from

$$
H_1 =
\begin{bmatrix}
1
\end{bmatrix},
$$

the next matrix is created with

$$
H_{2n} =
\begin{bmatrix}
H_n & H_n \\
H_n & -H_n
\end{bmatrix}.
$$

A Hadamard matrix satisfies

$$
H_n H_n^T = nI.
$$

To make the rows unit length, the matrix is normalized by multiplying every entry by
$1/\sqrt{n}$:

$$
Q_n = \frac{1}{\sqrt{n}} H_n .
$$

The normalized matrix has orthonormal rows:

$$
Q_n Q_n^T = I.
$$

This construction is used in numerical algorithms that need deterministic orthogonal transforms.

## Task

Implement `normalized_hadamard(n)`:

```python
def normalized_hadamard(n: int) -> np.ndarray:
    ...
```

The input $n$ is a positive power of two. Return the normalized Sylvester-Hadamard matrix as a NumPy array of shape $(n,n)$ with `float64` values.

Build the matrix using the recursive Sylvester construction. The returned matrix must match the deterministic construction and satisfy the orthogonality property.

## Example

```python
import numpy as np

H = normalized_hadamard(4)

# H is:
# [[ 0.5,  0.5,  0.5,  0.5],
#  [ 0.5, -0.5,  0.5, -0.5],
#  [ 0.5,  0.5, -0.5, -0.5],
#  [ 0.5, -0.5, -0.5,  0.5]]

np.allclose(H @ H.T, np.eye(4))
# True
```

## What the gate checks

The gate builds the reference matrix with the Sylvester recursion and compares the submitted matrix against it using the maximum absolute element error

$$
\max_{i,j} |A_{ij} - B_{ij}|.
$$

It also checks the orthogonality error

$$
\max_{i,j} |(QQ^T)_{ij} - I_{ij}|.
$$

Both errors must be at most $10^{-6}$.
