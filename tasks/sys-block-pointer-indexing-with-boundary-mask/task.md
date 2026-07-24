## Context

Triton kernels often process data in fixed-size tiles. A block pointer describes a rectangular region of memory using row and column offsets, but blocks near tensor boundaries may extend beyond valid elements.

A boundary mask prevents invalid memory accesses. For a matrix $A \in \mathbb{R}^{m \times n}$ and a requested tile starting at $(r_0, c_0)$ with size $B_m \times B_n$, a masked load returns

$$
T_{ij} =
\begin{cases}
A_{r_0+i,\ c_0+j}, & r_0+i < m \text{ and } c_0+j < n,\\
0, & \text{otherwise}.
\end{cases}
$$

This mirrors the behavior of a Triton masked load where out-of-range pointers are ignored and replaced with a default value.

## Task

Implement `block_pointer_gather(A, row_start, col_start, block_m, block_n)`.

The function takes:

- `A`: a 2-D NumPy array.
- `row_start`: the first row of the requested block.
- `col_start`: the first column of the requested block.
- `block_m`: the number of rows in the output tile.
- `block_n`: the number of columns in the output tile.

Return a NumPy array of shape $(block_m, block_n)$ containing the gathered tile. Elements outside the bounds of `A` must be filled with `0`. The output dtype should match `A.dtype`.

Do not clamp out-of-range indices to the nearest valid row or column. Boundary elements must use the mask rule.

## Example

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]], dtype=np.float32)

tile = block_pointer_gather(A, 1, 2, 3, 3)

# [[6., 0., 0.],
#  [0., 0., 0.],
#  [0., 0., 0.]]
```

## What the gate checks

The gate computes the expected masked tile directly from NumPy indexing and compares it with the submitted implementation.

The maximum absolute error

$$
\max_{i,j} |T_{ij}^{\mathrm{candidate}} - T_{ij}^{\mathrm{reference}}|
$$

must equal $0$. Implementations that read invalid elements, clamp indices, or return a differently shaped tile will fail.
