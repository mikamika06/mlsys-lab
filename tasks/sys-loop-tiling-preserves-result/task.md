## Context

Compilers and tensor systems such as XLA and TVM transform loop nests to improve
cache locality and hardware utilization. One common transformation is loop
tiling: splitting a large iteration space into smaller blocks while preserving
the mathematical result.

Matrix multiplication computes

$$
C_{ij} = \sum_{k=0}^{m-1} A_{ik} B_{kj}.
$$

A tiled implementation changes the order in which these partial sums are
accumulated. The result should remain numerically close to the untiled
computation while accessing smaller working sets.

For matrices $A \in \mathbb{R}^{n \times m}$ and
$B \in \mathbb{R}^{m \times p}$, the output is

$$
C \in \mathbb{R}^{n \times p}.
$$

The tile size controls the block dimensions used during the computation, but it
must not change the mathematical meaning of the operation.

## Task

Implement `tiled_matmul(A, B, tile)`:

```python
def tiled_matmul(A: np.ndarray, B: np.ndarray, tile: int) -> np.ndarray:
    ...
```

Return the matrix product of `A` and `B` using a tiled loop structure. Do not
call `np.matmul`, `@`, or `np.dot`. The input matrices contain `float64`
values and have compatible shapes.

The implementation should support matrix sizes that are not multiples of
`tile`. Partial tiles at the edges must be handled correctly.

## Example

```python
import numpy as np

A = np.array([[1., 2., 3.], [4., 5., 6.]])
B = np.array([[1., 0.], [0., 1.], [1., 1.]])

C = tiled_matmul(A, B, 2)

# C is approximately:
# [[4., 5.],
#  [10., 11.]]
```

## What the gate checks

The gate compares the implementation against the NumPy matrix multiplication
oracle. The maximum absolute error

$$
\max_{i,j} |C_{ij}^{candidate} - C_{ij}^{numpy}|
$$

must be less than $10^{-6}$ across several matrix sizes and tile sizes. A
correct tiled implementation passes while implementations that drop edge
blocks or incorrectly accumulate partial tiles fail.
