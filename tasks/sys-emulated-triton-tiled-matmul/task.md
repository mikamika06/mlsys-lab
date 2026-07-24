## Context

Triton kernels commonly divide matrix multiplication into tiles. A program instance computes one output tile and loads chunks of the input matrices. The important details are explicit accumulation across the reduction dimension and masking when a tile extends beyond the matrix boundary.

For matrices $A \in \mathbb{R}^{M \times K}$ and $B \in \mathbb{R}^{K \times N}$, matrix multiplication is

$$
C_{ij} = \sum_{k=0}^{K-1} A_{ik} B_{kj}.
$$

A tiled implementation chooses block sizes $T_M$, $T_N$, and $T_K$. For every output tile, it initializes an accumulator and repeatedly adds partial products:

$$
C_{tile} = \sum_{k\_tile} A_{tile} B_{tile}.
$$

When a tile reaches outside the valid matrix dimensions, the corresponding loads must be treated as zeros. This task emulates the algorithm on the CPU using NumPy arrays rather than launching a GPU kernel.

## Task

Implement `tiled_matmul(A, B, tile_m, tile_n, tile_k)`:

```python
def tiled_matmul(
    A: np.ndarray,
    B: np.ndarray,
    tile_m: int,
    tile_n: int,
    tile_k: int,
) -> np.ndarray:
    ...
```

The function receives a matrix $A$ of shape $(M, K)$ and a matrix $B$ of shape $(K, N)$ and returns a matrix $C$ of shape $(M, N)$.

Implement tiled accumulation:

- Iterate over output tiles of size `tile_m` by `tile_n`.
- Iterate over the reduction dimension in chunks of `tile_k`.
- Accumulate partial matrix products into an output tile.
- Handle partial tiles at the matrix edges correctly.

The output must be a `float64` NumPy array. Do not call `np.matmul`, `@`, or `np.dot` for the complete multiplication.

## Example

```python
import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
B = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float64)

C = tiled_matmul(A, B, 1, 2, 2)

# [[4. 5.]
#  [10. 11.]]
```

## What the gate checks

The gate compares the implementation against NumPy's matrix multiplication oracle. It computes the expected result using `np.matmul` and measures

$$
\max_{i,j} |C_{ij}^{candidate} - C_{ij}^{oracle}|.
$$

The returned `max_abs_err` must be less than $10^{-4}$. The cases include dimensions that are not multiples of the tile sizes, so missing edge masks or incomplete $K$ accumulation will fail.
