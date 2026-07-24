## Context

Sequence parallel attention can split the key and value sequence across multiple ranks. Each rank computes a partial attention state for the same query rows.

For one query row, let the attention logits on a rank be a vector $s_r$. The online softmax state stores:

$$
m_r = \max(s_r),
$$

$$
l_r = \sum_j \exp(s_{r,j} - m_r),
$$

and a weighted value accumulator

$$
o_r = \sum_j \exp(s_{r,j} - m_r) v_{r,j}.
$$

The rank states can be merged without keeping the original keys and values. The global maximum is

$$
M = \max_r(m_r).
$$

After rescaling each partial state, the final output is

$$
O =
\frac{\sum_r \exp(m_r - M)o_r}
{\sum_r \exp(m_r - M)l_r}.
$$

This merge operation is used by ring attention implementations where ranks exchange partial states instead of the full sequence.

## Task

Implement `reconstruct_output(states)`:

```python
def reconstruct_output(states):
    ...
```

`states` is a non-empty list of tuples `(m, l, o)` from different ranks.

Each item contains NumPy arrays:

- `m` has shape `(n,)` and contains per-query maxima.
- `l` has shape `(n,)` and contains per-query softmax normalization terms.
- `o` has shape `(n, d)` and contains the partial weighted value accumulators.

Return a NumPy array of shape `(n, d)` containing the merged attention output. The result must be `float64`.

Do not assume that all ranks have the same number of keys. Only the partial states are available.

## Example

```python
import numpy as np

states = [
    (
        np.array([2.0]),
        np.array([1.5]),
        np.array([[3.0, 6.0]])
    ),
    (
        np.array([1.0]),
        np.array([2.0]),
        np.array([[4.0, 8.0]])
    ),
]

out = reconstruct_output(states)
```

The function combines the two rank states using the rescaling formula instead of averaging the partial outputs.

## What the gate checks

The gate builds attention inputs with NumPy, splits keys and values across ranks, computes the true attention states from the split inputs, and checks the reconstruction against the NumPy attention oracle.

The reported metric is

$$
\max_{i,j}|O_{i,j}^{candidate} - O_{i,j}^{oracle}|.
$$

The value must be less than $10^{-6}$.
