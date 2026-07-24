## Context

Structured sparsity formats reduce storage by keeping only nonzero weights and a small amount of metadata. In a common 2:4 sparsity pattern, every group of four weights contains exactly two stored values. The compressed representation stores the two surviving values and two indices describing their original locations.

For a dense weight matrix $W \in \mathbb{R}^{m \times n}$, a binary mask $M$ describes which entries survive compression. The reconstructed tensor is

$$
\hat{W} = W \odot M ,
$$

where $\odot$ denotes elementwise multiplication. The compression step removes zeros and stores only the values at positions where $M_{ij}=1$.

The index metadata uses two bits per kept value because each group of four positions has possible locations $0,1,2,3$. For a row flattened into groups

$$
(x_0,x_1,x_2,x_3), (x_4,x_5,x_6,x_7), \ldots
$$

the compressed values contain the two retained elements from each group, and the indices contain their offsets inside the corresponding group.

## Task

Implement `reconstruct_24(values, indices, shape)`:

```python
def reconstruct_24(values: np.ndarray, indices: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    ...
```

The inputs are:

- `values`: a 2-D array with shape `(rows, cols // 2)` containing the nonzero values in row-major group order.
- `indices`: a 2-D integer array with the same shape as `values`. Each entry is the offset `$0 \le k < 4$` of the value inside its 4-element group.
- `shape`: the original dense matrix shape `(rows, cols)`.

Return a `float32` dense matrix with the stored values placed at their original locations and all other entries equal to zero.

The number of columns is always divisible by $4$. Each group of four consecutive columns contributes exactly two values.

## Example

```python
import numpy as np

values = np.array([[1.5, 2.5, 3.0, 4.0]], dtype=np.float32)
indices = np.array([[0, 3, 1, 2]], dtype=np.int8)

W = reconstruct_24(values, indices, (1, 8))

# W is:
# [[1.5, 0. , 0. , 2.5, 0. , 3.0, 4.0, 0. ]]
```

## What the gate checks

The gate compares the returned dense tensor against a NumPy oracle that expands the compressed representation by computing each group's original positions and scattering the stored values.

The `exact_match` score must be exactly $1.0$. Any incorrect placement of values, wrong shape handling, or incorrect zeros causes the gate to fail.
