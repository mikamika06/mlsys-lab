## Context

Production attention implementations often process key/value tokens in chunks to reduce memory usage. The softmax denominator cannot be computed independently for each chunk because the normalization statistics must be merged.

For a query vector $q$ and key matrix $K$, attention scores are

$$s_i = q^\top k_i.$$

The softmax weights are

$$p_i = \frac{e^{s_i}}{\sum_j e^{s_j}}.$$

A numerically stable implementation tracks a running maximum $m$ and normalization accumulator $l$. When a new chunk has maximum score $m_{\mathrm{new}}$, previous statistics are rescaled:

$$
l' = e^{m-m_{\mathrm{new}}} l + \sum_i e^{s_i-m_{\mathrm{new}}}.
$$

The accumulated output is updated using the same rescaling factor. Resetting $m$ and $l$ for every chunk computes separately normalized pieces and produces incorrect attention values.

## Task

Implement `chunked_attention(q, chunks)`.

The argument `q` is a list of floats with shape $(d,)$.
`chunks` is a list of pairs `(K_chunk, V_chunk)` where `K_chunk` has shape
$(n_i, d)$ and `V_chunk` has shape $(n_i, h)$.

Return a list of shape $(h,)$ containing the attention output over all
keys and values from all chunks. Use a numerically stable running softmax
algorithm. The running maximum and normalization state must be carried from one
chunk to the next. The result must be `float64`.

## Example

```python

q = [1.0, 0.0]
chunks = [
    ([[1.0, 0.0], [0.0, 1.0]], [[1.0], [2.0]]),
    ([[10.0, 0.0]], [[3.0]]),
]

y = chunked_attention(q, chunks)
```

The output is the same as applying softmax once over all three scores, not as
averaging the outputs of two independent chunk softmax operations.

## What the gate checks

The gate computes a Python reference implementation that performs the full stable
online softmax merge with running state in float64. The candidate result is
compared with the oracle using

$$\max_i |y_i-\hat{y}_i|.$$

The maximum absolute error must be below $10^{-5}$. An implementation that
resets the running maximum and normalization at every chunk produces a larger
error on the tested cases.
