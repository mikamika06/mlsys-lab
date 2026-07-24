## Context

The QR decomposition factors a matrix $A \in \mathbb{R}^{m \times n}$ into an orthonormal matrix $Q$
and an upper triangular matrix $R$:

$$
A = QR,
$$

where the columns of $Q$ satisfy

$$
Q^\top Q = I.
$$

Classical Gram-Schmidt constructs the columns of $Q$ one at a time. For column $a_j$ of $A$,
subtract the projections onto the previously computed orthonormal columns:

$$
u_j = a_j - \sum_{i=1}^{j-1} (q_i^\top a_j)q_i .
$$

The new orthonormal column is

$$
q_j = \frac{u_j}{\lVert u_j \rVert_2}.
$$

The entries of $R$ contain the projection coefficients:

$$
R_{ij} = q_i^\top a_j,
$$

including the diagonal term

$$
R_{jj} = \lVert u_j \rVert_2 .
$$

## Task

Implement `classical_gram_schmidt_qr(A)`:

```python
def classical_gram_schmidt_qr(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ...
```

The function receives a real-valued NumPy array of shape $(m, n)$ with $m \ge n$ and returns
`(Q, R)` where `Q` has shape $(m, n)$ and `R` has shape $(n, n)$.

Implement classical Gram-Schmidt directly. Do not call `np.linalg.qr`. The result should be a
floating point QR decomposition of the input matrix.

## Example

```python
import numpy as np

A = np.array([
    [1.0, 1.0],
    [1.0, 0.0],
    [0.0, 1.0],
])

Q, R = classical_gram_schmidt_qr(A)

# Q @ R reconstructs A approximately.
```

## What the gate checks

The grader computes a QR decomposition using NumPy as a numerical oracle and compares the submitted
$Q$ and $R$ matrices after resolving the possible sign ambiguity of QR factors.

The reported metric is

$$
\max_{i,j} |X_{ij} - X^{\mathrm{ref}}_{ij}|,
$$

over both returned factors. The value must be below $10^{-9}$.

The grader also rejects implementations that call `np.linalg.qr`.
