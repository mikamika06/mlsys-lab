## Context

In a GPU programming model, work is distributed over many lightweight threads organized into blocks. Each thread can query its unique identifiers:

* `$\\texttt{threadIdx}$` – index of the thread within its block  
* `$\\texttt{blockIdx}$` – index of the block within the grid  
* `$\\texttt{blockDim}$` – size (number of threads) of each block  

The global linear index that uniquely identifies a thread across all blocks is obtained by

$$
g_{\mathrm{id}} = \\texttt{blockIdx} \\times \\texttt{blockDim} + \\texttt{threadIdx}.
$$

If the grid contains `$\\texttt{gridDim}$` blocks, each of size `$\\texttt{blockDim}$`, then the total number of threads is `$N = \\texttt{gridDim}\\times\\texttt{blockDim}``.

## Task

Implement a function that returns a flat list containing the global indices for every thread in a 1‑D grid launch.

```python
def launch_indices(block_dim: int, grid_dim: int) -> list[int]:
    ...
```

* `block_dim` – number of threads per block (`int > 0`).  
* `grid_dim`  – number of blocks in the grid (`int > 0`).  

The function must return a list of shape `(block_dim * grid_dim,)`, dtype int, containing all values
$g_{\mathrm{id}} = i\\times\\texttt{blockDim} + j$ for every pair $(i,j)$ with  
$i \\in [0, \\texttt{gridDim}-1]$ and $j \\in [0, \\texttt{blockDim}-1]$.  
The resulting array should be in increasing order: the first element corresponds to block 0, thread 0; the last element to block `grid_dim-1`, thread `block_dim-1`.

## Example

```python
idx = launch_indices(3, 4)
print(idx)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
```

## What the gate checks

The grader verifies that the returned array is exactly equal to a reference computed by the same formula, using Python's `arange`. No other constraints are imposed; however, the output must have type int and match shape `(block_dim * grid_dim,)`.
