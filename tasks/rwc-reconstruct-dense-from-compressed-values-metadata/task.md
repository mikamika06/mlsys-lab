## Context

Many inference systems compress neural network weights using structured sparsity. A common pattern is $2:4$ compression: every group of four weights stores exactly two values and metadata describing which two positions were kept.

For a dense vector $x \in \mathbb{R}^{4m}$, each block

$$
(x_{4k}, x_{4k+1}, x_{4k+2}, x_{4k+3})
$$

is represented by two values and two 2-bit indices. Each index $p \in \{0,1,2,3\}$ identifies the original location inside the four-element block. The two indices are packed into one byte:

$$
\text{metadata}_k = p_0 + 4p_1 .
$$

The compressed representation stores the kept values in block order and stores one metadata byte per block. Reconstruction reverses the compression by scattering each stored value back to its original position.

## Task

Implement `reconstruct_dense(values, metadata, shape)`.

The inputs are:

- `values`: a 1-D NumPy array containing the two kept values from every block, in block order.
- `metadata`: a 1-D NumPy array of unsigned bytes. Each byte contains two 2-bit positions. The low two bits are the first position and bits 2-3 are the second position.
- `shape`: the original 2-D dense weight shape. The total number of elements is divisible by $4$.

Return a tuple `(dense, positions)`:

- `dense` is a NumPy array with dtype `float64` and the requested shape.
- `positions` is a 1-D NumPy array containing the recovered positions inside each 4-element block in the same order as the metadata expansion.

The reconstruction should scatter the values into the correct locations and leave all other entries as zero.

## Example

```python
import numpy as np

values = np.array([1.5, -2.0, 3.0, 4.0])
metadata = np.array([0b00011000, 0b00000101], dtype=np.uint8)

dense, positions = reconstruct_dense(values, metadata, (2, 4))
```

The first metadata byte decodes to positions $0$ and $2$. The function should place the first two values into those locations and leave the other locations in the block as zero.

## What the gate checks

The gate creates dense weights with a NumPy compression oracle, then reconstructs the compressed representation.

`max_abs_err` measures the maximum absolute difference between the reconstructed dense matrix and the oracle-compressed dense matrix. It must satisfy $0$ error.

`positions_exact` checks that the recovered expanded metadata positions exactly match the oracle positions. The implementation must decode both 2-bit fields from every metadata byte.
