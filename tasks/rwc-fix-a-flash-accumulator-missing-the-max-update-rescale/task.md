## Context

Flash attention implementations avoid materializing the full attention matrix by
streaming blocks of keys and values while maintaining a running softmax
normalizer. For a query vector $q$, keys $K$, and values $V$, the output is

$$
o = \frac{\sum_j \exp(q^\top k_j - m) v_j}
{\sum_j \exp(q^\top k_j - m)},
$$

where $m$ is the maximum score used for numerical stability.

During streaming, the implementation keeps a running maximum $m$ and two
accumulators:

$$
s = \sum_j \exp(q^\top k_j - m),
$$

$$
a = \sum_j \exp(q^\top k_j - m)v_j .
$$

When a new block has a larger maximum score, the previous accumulator values were
scaled using the old maximum. They must be converted to the new scale:

$$
s_{\text{old}} \leftarrow s_{\text{old}}\exp(m_{\text{old}} - m_{\text{new}})
$$

$$
a_{\text{old}} \leftarrow a_{\text{old}}\exp(m_{\text{old}} - m_{\text{new}}).
$$

Missing this rescale causes incorrect outputs when later blocks contain larger
scores.

## Task

Implement `flash_attention_accumulate(q, K, V, block_size)`.

The function receives:

- `q`: a 1-D NumPy array of shape $(d,)$.
- `K`: a 2-D NumPy array of shape $(n, d)$ containing keys.
- `V`: a 2-D NumPy array of shape $(n, d_v)$ containing values.
- `block_size`: the number of rows processed per streaming block.

Return the final attention output as a 1-D NumPy array of shape $(d_v,)$.

Process the keys and values in consecutive blocks. Maintain a running maximum
and accumulators, and apply the max-update rescale whenever the running maximum
increases. The implementation should return `float64` values.

## Example

```python
import numpy as np

q = np.array([1.0, 0.0])
K = np.array([[1.0, 0.0], [0.0, 1.0]])
V = np.array([[2.0], [4.0]])

out = flash_attention_accumulate(q, K, V, 1)
# approximately array([2.53788284])
```

## What the gate checks

The gate computes a NumPy reference by evaluating the naive softmax attention
formula directly:

$$
\mathrm{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}.
$$

The submitted implementation is tested on inputs where a later block increases
the running maximum. The returned vector must have maximum absolute error at most
the required threshold compared with the NumPy oracle. An implementation that
omits the accumulator rescale will produce a numerically different result.
