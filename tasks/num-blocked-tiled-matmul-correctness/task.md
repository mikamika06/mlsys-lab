## Context

Matrix multiplication is a core operation in numerical computing. For matrices
$A \in \mathbb{R}^{m \times k}$ and $B \in \mathbb{R}^{k \times n}$, the result
$C \in \mathbb{R}^{m \times n}$ is defined by

$$
C_{ij} = \sum_{t=0}^{k-1} A_{it} B_{tj}.
$$

A direct implementation computes one output element at a time. Blocked matrix
multiplication improves memory locality by splitting the computation into tiles.
For a block size $B_s$, the matrix is divided into smaller regions and the
algorithm accumulates partial products:

$$
C_{I,J} \mathrel{+}= A_{I,K} B_{K,J}.
$$

The final result must be identical to the mathematical matrix product. Blocking
changes the order of loops, not the computation being performed.

## Task

Implement `blocked_matmul(A, B, block_size)`:

```python
def blocked_matmul(A: np.ndarray, B: np.ndarray, block_size: int) -> np.ndarray:
    ...
```

The function receives two 2-D NumPy arrays where `A.shape[1] == B.shape[0]`.
Return a new `float64` array containing $A B$.

Use blocked/tiled multiplication with three block loops over the output row,
output column, and reduction dimensions. Within each tile, use NumPy operations
for the small matrix multiplication and accumulate the partial result.

The function must support matrix dimensions that are not multiples of
`block_size`.

## Example

```python
import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
B = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float64)

C = blocked_matmul(A, B, 2)

# C is:
# [[4. 5.]
#  [10. 11.]]
```

## What the gate checks

The gate computes a reference answer using NumPy's matrix multiplication and
compares it with `blocked_matmul`.

The reported metric is the maximum absolute error

$$
\max_{i,j} |C_{ij}^{candidate} - C_{ij}^{reference}|.
$$

The value must be less than $10^{-6}$. The cases include rectangular matrices
and sizes that require partial edge tiles.
