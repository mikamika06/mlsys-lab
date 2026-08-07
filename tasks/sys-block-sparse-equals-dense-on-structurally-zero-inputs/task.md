## Context

Attention computes a weighted combination of value vectors. Given query matrix
$Q \in \mathbb{R}^{n \times d}$, key matrix $K \in \mathbb{R}^{n \times d}$,
and value matrix $V \in \mathbb{R}^{n \times d_v}$, the dense attention output is

$$
Y = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V .
$$

For long sequences, many attention variants use block sparsity. The sequence is
split into blocks of size $b$, and a block pattern decides which query blocks
can interact with which key blocks.

In this task, the input is guaranteed to have structural zeros outside the
allowed block pattern. Therefore, skipping those blocks must produce the same
numeric result as dense computation. The modeled dense score work is
proportional to

$$
n^2 d ,
$$

while block-sparse score work is proportional to the number of active block
pairs:

$$
N_{\mathrm{active}} b^2 d .
$$

The savings ratio is

$$
\frac{n^2 d}{N_{\mathrm{active}} b^2 d}.
$$

## Task

Implement `block_sparse_attention(Q, K, V, block_mask, block_size)`.

The inputs are list:

```python
def block_sparse_attention(Q, K, V, block_mask, block_size):
    ...
```

`Q`, `K`, and `V` have matching row counts. `block_mask` is a square boolean
matrix over sequence blocks. A `True` entry means that query block may use the
corresponding key block.

The function must:

1. Compute the attention output using only active blocks.
2. Return the output as `float64`.
3. Return the modeled FLOP savings ratio as a Python `float`.

The gate inputs are structurally zero: every query-key score contribution from a
disabled block pair is zero, so the sparse computation must match dense
computation.

Do not use external libraries other than Python.

## Example

```python

Q = [[1., 0.], [0., 1.], [1., 1.], [2., 1.]]
K = [[0 for _ in row] for row in Q]
V = [[1., 0.], [0., 1.], [1., 1.], [2., 2.]]
mask = [[True, False], [False, True]]

Y, ratio = block_sparse_attention(Q, K, V, mask, 2)
```

The function computes only the two diagonal blocks. The returned matrix must
match the dense reference for the same structurally zero input.

## What the gate checks

The grader builds several structured inputs and computes a dense Python oracle
inside the checker. The reported output must satisfy
$\max_i |Y_i-\hat{Y}_i| < 10^{-6}$.

The second metric verifies the returned modeled FLOP savings ratio against the
ratio computed from the active block count. A solution that only produces the
dense result but ignores the sparse work model will fail this check.
