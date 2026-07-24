## Context

A structured $2:4$ sparse matrix format stores weights in groups of four values. Each group keeps exactly two nonzero values and records metadata describing which two positions are stored.

For a weight matrix $W \in \mathbb{R}^{m \times n}$ with $2:4$ sparsity, the dense FP16 storage cost is

$$S_{\mathrm{dense}} = 2mn \text{ bytes},$$

because each FP16 value uses $2$ bytes.

The compressed format stores the nonzero FP16 values and $2$ bits of metadata per group of four values:

$$S_{\mathrm{packed}} = 2 \cdot \mathrm{nnz} + \left\lceil \frac{2G}{8} \right\rceil,$$

where $\mathrm{nnz}$ is the number of nonzero values and $G$ is the number of four-value groups.

The storage reduction ratio is

$$R = \frac{S_{\mathrm{dense}}}{S_{\mathrm{packed}}}.$$

A true $2:4$ matrix has

$$\frac{\mathrm{nnz}}{mn} = 0.5.$$

## Task

Implement `measure_24_reduction(W)`:

```python
def measure_24_reduction(W):
    ...
```

`W` is a 2-D NumPy array containing FP16 weights. The shape is guaranteed to be divisible into groups of four along the last dimension, and every group contains exactly two nonzero values.

Return a tuple:

```python
(density, packed_bytes)
```

where:

- `density` is the nonzero fraction as a Python float.
- `packed_bytes` is the number of bytes required by the $2:4$ packed representation as an integer.

Do not return the compression ratio. The grader computes that from your reported packed size.

## Example

```python
import numpy as np

W = np.array(
    [[1, 0, 2, 0],
     [0, 3, 0, 4]],
    dtype=np.float16,
)

density, packed_bytes = measure_24_reduction(W)
# density == 0.5
# packed_bytes == 9
```

There are $8$ dense FP16 values using $16$ bytes. The compressed representation stores $4$ nonzero FP16 values using $8$ bytes plus $1$ byte of metadata.

## What the gate checks

The grader recomputes the $2:4$ storage calculation from the NumPy input and compares your returned values with the oracle result.

The `density` gate requires exact agreement with the oracle. The `size_ratio` gate checks

$$\frac{S_{\mathrm{dense}}}{S_{\mathrm{packed}}}$$

against the oracle-computed ratio with an error tolerance of $10^{-6}$.

Returning the dense byte count, ignoring metadata, or using an incorrect nonzero count will fail.
