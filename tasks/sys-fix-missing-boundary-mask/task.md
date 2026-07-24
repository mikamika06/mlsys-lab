## Context

Triton kernels often process arrays in fixed-size blocks. A program instance loads a
block of elements, but the final block may extend beyond the end of the input array.

For an input vector $x \in \mathbb{R}^{n}$ and a block size $B$, the number of
blocks is

$$
m = \left\lceil \frac{n}{B} \right\rceil .
$$

A block reduction should only include valid indices. For block $k$, the valid
indices are

$$
kB \le i < \min((k+1)B, n).
$$

The equivalent masked load in a kernel uses a predicate that disables lanes where
the index is outside the input range. Without this mask, the last block can read
past the end of the array and introduce invalid values into the reduction.

## Task

Implement `masked_block_sum(x, block_size)`:

```python
def masked_block_sum(x: np.ndarray, block_size: int) -> np.ndarray:
    ...
```

The function receives a 1-D NumPy array `x` and a positive integer block size.
Return a 1-D NumPy array containing the sum of every consecutive block.

The output length must be

$$
\left\lceil \frac{\mathrm{len}(x)}{\mathrm{block\_size}} \right\rceil .
$$

Only elements inside `x` may contribute to a block sum. In particular, the final
block must ignore lanes whose indices are greater than or equal to `len(x)`.
Return results as `float64`.

## Example

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
y = masked_block_sum(x, 4)

# y is:
# [10. 5.]
```

The first block contains four valid values. The second block contains only the
last value, because the remaining positions are outside the array boundary.

## What the gate checks

The gate computes a NumPy oracle by explicitly summing only valid array indices
for each block. The submitted implementation is compared against this reference
using global relative error

$$
\mathrm{rel\_err} =
\frac{\lVert y_{\mathrm{candidate}} - y_{\mathrm{reference}} \rVert_2}
{\lVert y_{\mathrm{reference}} \rVert_2 + 10^{-12}} .
$$

A missing boundary mask changes the final block values and fails the
$\mathrm{rel\_err} \le 10^{-6}$ requirement.
