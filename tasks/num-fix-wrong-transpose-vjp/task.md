## Context

A vector-Jacobian product (VJP) applies the chain rule backwards through a
primitive operation. For matrix multiplication

$$
Y = A B,
$$

with an upstream gradient $G = \frac{\partial L}{\partial Y}$, the VJP must
return gradients for both inputs:

$$
\frac{\partial L}{\partial A} = G B^\top
$$

and

$$
\frac{\partial L}{\partial B} = A^\top G .
$$

A common implementation bug is using $B$ instead of $B^\top$ when computing the
gradient with respect to $A$. This produces a shape-compatible result for some
cases but violates the chain rule.

## Task

Implement `matmul_vjp(A, B, G)`:

```python
def matmul_vjp(A: list[list[float]], B: list[list[float]], G: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    ...
```

The inputs are list of lists of floats where `A` has shape $(m, k)$,
`B` has shape $(k, n)$, and `G` has shape $(m, n)$.

Return a tuple `(dA, dB)` containing the VJP gradients for `A` and `B`. The
returned arrays must use Python matrix operations and have the same shapes as
`A` and `B`.

## Example

```python

A = [[1.0, 2.0], [3.0, 4.0]]
B = [[5.0, 6.0], [7.0, 8.0]]
G = [[1.0] * 2 for _ in range(2)]

dA, dB = matmul_vjp(A, B, G)

# dA = G @ B.T
# dB = A.T @ G
```

## What the gate checks

The gate builds a scalar function

$$
L(A,B) = \sum_{i,j} (AB)_{ij}G_{ij}
$$

and computes the reference gradients using central finite differences. The
submitted VJP outputs are compared against this numerical oracle. The maximum
absolute error must satisfy

$$
\max |x_{\mathrm{submitted}} - x_{\mathrm{finite\ difference}}| < 10^{-5}.
$$
