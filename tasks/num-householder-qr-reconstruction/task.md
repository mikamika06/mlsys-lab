## Context

The QR decomposition factors a matrix $A \in \mathbb{R}^{m \times n}$ into an orthogonal
matrix $Q$ and an upper triangular matrix $R$:

$$
A = QR.
$$

Householder QR constructs this factorization using reflections that remove entries
below each diagonal pivot. A Householder reflector has the form

$$
H = I - 2\frac{vv^\top}{v^\top v},
$$

where $v$ is selected so that multiplying by $H$ transforms a vector into one
with zeros below its first component.

Applying a sequence of reflectors gives

$$
R = H_k \dots H_2 H_1 A.
$$

The accumulated orthogonal factor is the product of the same reflectors in reverse
order:

$$
Q = H_1 H_2 \dots H_k.
$$

A correct implementation reconstructs the original matrix:

$$
QR \approx A.
$$

## Task

Implement `householder_qr_reconstruct(A)`:

```python
def householder_qr_reconstruct(A):
    ...
```

The function takes a two-dimensional NumPy array and returns `(Q, R)`.

Requirements:

- Use Householder reflections to compute the QR factorization.
- Return `Q` with shape $(m,m)$.
- Return `R` with shape $(m,n)$.
- Do not modify the input array.
- The product `Q @ R` must reconstruct the input matrix within floating point
  precision.

## Example

```python
import numpy as np

A = np.array([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
])

Q, R = householder_qr_reconstruct(A)

print(np.allclose(Q @ R, A))
# True
```

## What the gate checks

The gate builds a reconstruction reference using NumPy's QR implementation.
It also verifies that the returned factors have the required full shapes.

The reported metric is `max_abs_err`, the largest absolute reconstruction error.
Invalid factor shapes are scored as failure.

The gate passes when

$$
\mathrm{max\_abs\_err} < 10^{-10}.
$$
