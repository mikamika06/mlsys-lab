## Context

In automatic differentiation the backward pass must undo every shape change that
broadcasting introduced in the forward pass.  When a row-vector $b \in \mathbb{R}^{m}$
is added to each row of a matrix $A \in \mathbb{R}^{n \times m}$, Python broadcasts
$b$ to shape $(n, m)$ by implicit replication:

$$C_{ij} = A_{ij} + b_{j}.$$

The gradient of a scalar loss $\mathcal{L}$ with respect to $b$ therefore requires
a sum over the broadcast (row) dimension:

$$\frac{\partial \mathcal{L}}{\partial b_{j}}
  = \sum_{i=1}^{n} \frac{\partial \mathcal{L}}{\partial C_{ij}}.$$

In vector form, $\nabla_{b}\,\mathcal{L} = (\nabla_{C}\,\mathcal{L})^{\!\top}\,
\mathbf{1}_{n}$, which in Python is `[sum(col) for col in zip(*dc)]`. Returning the raw upstream
gradient `dc` without reducing it is a common autograd bug: the result has the wrong
shape $(n, m)$ instead of $(m,)$ and every downstream parameter update is corrupted.

## Task

The function `broadcast_add(a, b)` computes the forward pass $C = A + b$ (with $b$
broadcast across rows of $A$) and returns a `(c, backward)` pair.
`backward(dc)` must return `(da, db)` — the gradients of
$\mathcal{L} = \sum_{ij} dc_{ij}\, C_{ij}$ with respect to $a$ and $b$.

The provided implementation is **buggy**: the `backward` function does not sum
over the broadcast axis when computing `db`.  Fix it so that:

1. `da` has shape $(n, m)$ and equals $dc$.
2. `db` has shape $(m,)$ and equals `[sum(col) for col in zip(*dc)]`.

Do **not** change the forward pass.

## Example

```python
a = [[1.0, 2.0],
              [3.0, 4.0]]
b = [10.0, 20.0]
c, backward = broadcast_add(a, b)
# c = [[11., 22.],
#       [13., 24.]]
dc = [[1.0 for _ in row] for row in c]
da, db = backward(dc)
# da = [[1., 1.],
#        [1., 1.]]       ← shape (2, 2)
# db = [2., 2.]           ← shape (2,), each entry is sum of dc column
```

## What the gate checks

Central finite differences with step $\epsilon = 10^{-5}$ compute the numerical
gradient of $\mathcal{L} = \sum dc \odot C$ with respect to both $a$ and $b$, using
the known-correct forward $C = A + b$.  The maximum absolute errors

$$\max_{i,j}\bigl|\texttt{da\_student}_{ij} - \texttt{da\_num}_{ij}\bigr|
  \quad\text{and}\quad
  \max_{j}\bigl|\texttt{db\_student}_{j} - \texttt{db\_num}_{j}\bigr|$$

must both be below $10^{-5}$.
A shape mismatch (the buggy `db` is $(n,m)$ instead of $(m,)$) is caught
immediately; even if shapes happen to broadcast, the numeric values disagree and
the gate fails.
