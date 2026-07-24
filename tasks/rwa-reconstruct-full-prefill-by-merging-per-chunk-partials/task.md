## Context

Large attention computations can be split into chunks to reduce memory usage. Each chunk computes partial attention statistics instead of the final output.

For a query attending over keys with logits $x_i$, the attention weights are

$$
p_i = \frac{e^{x_i}}{\sum_j e^{x_j}} .
$$

A numerically stable chunk stores three values:

- $m$, the maximum logit in the chunk.
- $l$, the sum of exponentials shifted by the chunk maximum:

$$
l = \sum_i e^{x_i-m}.
$$

- $o$, the weighted value accumulation:

$$
o = \sum_i e^{x_i-m} v_i .
$$

To reconstruct the full attention output from multiple chunks, the partials must be merged using log-sum-exp. For chunk statistics $(m_k,l_k,o_k)$, compute

$$
M = \max_k m_k ,
$$

then combine the normalizers:

$$
L = \sum_k l_k e^{m_k-M}.
$$

The final output is

$$
O = \frac{\sum_k o_k e^{m_k-M}}{L}.
$$

This is the same merge rule used by production attention kernels when combining independently computed chunks.

## Task

Implement `merge_chunk_partials(ms, ls, os)`:

```python
def merge_chunk_partials(ms, ls, os):
    ...
```

The inputs are:

- `ms`: a 1-D NumPy array of chunk maxima with shape `(chunks,)`.
- `ls`: a 1-D NumPy array of chunk exponential sums with shape `(chunks,)`.
- `os`: a 2-D NumPy array of chunk weighted accumulations with shape `(chunks, d)`.

Return a 1-D NumPy array of length `d` containing the reconstructed full attention output.

Use NumPy operations and keep the computation numerically stable. The result should use `float64` arithmetic.

## Example

```python
import numpy as np

ms = np.array([2.0, 1.0])
ls = np.array([3.0, 4.0])
os = np.array([[6.0, 3.0], [4.0, 8.0]])

out = merge_chunk_partials(ms, ls, os)
```

The function first finds the global maximum, rescales each chunk contribution, merges the accumulations, and divides by the merged normalizer.

## What the gate checks

The gate creates attention logits and values in NumPy, splits them into chunks, and computes chunk statistics. It independently computes the full attention output from the original logits in float64.

The returned output must satisfy

$$
\max_i |O_i - \hat{O}_i| < 10^{-5},
$$

where $O$ is the NumPy oracle result and $\hat{O}$ is the submitted implementation result.
