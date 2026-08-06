## Context

Attention computes a weighted combination of value vectors. Given queries $Q$, keys $K$, and values $V$, the usual scaled dot-product attention is

$$
O = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V .
$$

For large language models, materializing the full score matrix $QK^\top$ can require too much memory. FlashAttention uses an online softmax recurrence that processes keys and values in blocks while maintaining the same result.

For one query row, maintain a running maximum $m$, a running normalization term $l$, and an accumulated output $o$. For a new block of scores $s$:

$$
m_{\text{new}} = \max(m, \max(s)),
$$

$$
l_{\text{new}} = l e^{m-m_{\text{new}}} + \sum_j e^{s_j-m_{\text{new}}},
$$

$$
o_{\text{new}} =
\frac{
l e^{m-m_{\text{new}}} o + \sum_j e^{s_j-m_{\text{new}}}v_j
}{
l_{\text{new}}
}.
$$

This recurrence avoids storing the complete attention matrix while preserving numerical stability.

## Task

Implement `streaming_softmax_attention(Q, K, V, block_size)`:

```python
def streaming_softmax_attention(Q, K, V, block_size):
    ...
```

The inputs are:

- `Q` with shape $(n, d)$ containing query vectors.
- `K` with shape $(n, d)$ containing key vectors.
- `V` with shape $(n, d_v)$ containing value vectors.
- `block_size` controlling how many key/value rows are processed per streaming step.

Return the attention output with shape $(n, d_v)$ and dtype `float64`.

Use the streaming softmax recurrence. Do not construct the full $n \times n$ attention score matrix.

## Example

```python

Q = [[1.0, 0.0]]
K = [[1.0, 0.0], [0.0, 1.0]]
V = [[2.0], [4.0]]

out = streaming_softmax_attention(Q, K, V, 1)
```

The result is the same as computing

$$
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{2}}\right)V
$$

directly.

## What the gate checks

The gate builds deterministic query, key, and value arrays and computes a Python oracle using the full scaled dot-product attention expression. The candidate implementation is compared against that oracle.

The reported metric is `max_abs_err`, the largest absolute difference between the candidate output and the oracle output. The value must satisfy

$$
\max_i |O_i-\hat{O}_i| < 10^{-5}.
$$

A solution that only concatenates blocks or applies softmax independently to each block will fail because it does not preserve the running normalization across blocks.
