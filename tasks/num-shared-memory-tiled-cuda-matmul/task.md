## Context

Matrix multiplication computes a matrix product $C = AB$. For matrices
$A \in \mathbb{R}^{m \times k}$ and $B \in \mathbb{R}^{k \times n}$,

$$
C_{ij} = \sum_{t=0}^{k-1} A_{it} B_{tj}.
$$

A naive CUDA kernel assigns one thread to each output element. During the
reduction over $k$, every multiply-add may read values directly from global
memory. A shared-memory tiled kernel reduces this traffic by loading tiles of
$A$ and $B$ into shared memory and reusing those values.

For a tile width $T$, each output block processes the reduction dimension in
chunks. In every chunk, the block loads the valid part of an $T \times T$ tile
of $A$ and an $T \times T$ tile of $B$. Threads then compute partial sums from
the staged data.

This task models the global memory behavior of a tiled CUDA GEMM kernel. The
implementation is written in Python, but the access count represents the CUDA
kernel's global reads. The numeric result must match the NumPy oracle.

## Task

Implement `tiled_cuda_matmul(A, B, tile_size)`.

Arguments:

- `A`: a 2-D NumPy array with shape $(m, k)$ and dtype `float64`.
- `B`: a 2-D NumPy array with shape $(k, n)$ and dtype `float64`.
- `tile_size`: the shared-memory tile width $T$.

Return:

```python
(C, global_loads)
```

where:

- `C` is the matrix product with dtype `float64`.
- `global_loads` is the modeled number of global memory reads.

Model the tiled kernel by loading each valid element of the current $A$ and $B$
tiles once per output block and reduction tile. Values inside shared memory are
then reused for accumulation.

## Example

```python
import numpy as np

A = np.array([[1., 2.], [3., 4.]])
B = np.array([[5., 6.], [7., 8.]])

C, loads = tiled_cuda_matmul(A, B, 2)

# C:
# [[19. 22.]
#  [43. 50.]]
```

## What the gate checks

The gate computes $C_{\mathrm{ref}} = AB$ using NumPy and checks the maximum
absolute error of the returned matrix.

It also computes the expected tiled global-read count from the tile-loading
algorithm and compares the returned count. The reported
$\mathrm{modeled\_access\_count}$ metric is

$$
\frac{\mathrm{returned\ global\ loads}}
{\mathrm{oracle\ tiled\ global\ loads}} .
$$

The value must satisfy $\mathrm{modeled\_access\_count} \le 1$. A naive kernel
that reloads $A$ and $B$ for every multiply-add has a larger access count and
fails this gate.
