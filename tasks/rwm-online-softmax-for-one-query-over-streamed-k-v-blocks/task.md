## Context

Attention for one query vector $q$ over keys $K$ and values $V$ is normally written as

$$
\mathrm{Attention}(q,K,V) =
\frac{\sum_i \exp(s_i) V_i}{\sum_i \exp(s_i)},
$$

where

$$
s_i = \frac{q^\top K_i}{\sqrt{d}} .
$$

Computing all scores and then applying softmax requires keeping every score in memory. Production attention kernels avoid this by processing keys and values in blocks.

The online softmax algorithm keeps a running maximum $m$, a running normalization value $\ell$, and an accumulated output $a$. When a new block has scores $x$, it updates:

$$
m_{\mathrm{new}} = \max(m, \max(x)),
$$

$$
\ell_{\mathrm{new}} =
\exp(m - m_{\mathrm{new}})\ell +
\sum_j \exp(x_j - m_{\mathrm{new}}),
$$

$$
a_{\mathrm{new}} =
\frac{
\exp(m - m_{\mathrm{new}})\ell a +
\sum_j \exp(x_j - m_{\mathrm{new}})V_j
}{
\ell_{\mathrm{new}}
}.
$$

This rescaling allows blocks to be streamed without numerical overflow from large softmax logits.

## Task

Implement `online_attention(q, K_blocks, V_blocks)`.

The function receives:

- `q`: a list of floats of shape $(d,)$.
- `K_blocks`: a list of list of lists of floats. Each block has shape $(b_i, d)$.
- `V_blocks`: a list of list of lists of floats. Each block has shape $(b_i, d_v)$.

The function must process blocks in order and return the single attention output vector of shape $(d_v,)$.

Use the online softmax algorithm. Do not concatenate all key or value blocks into one full matrix. The returned array must use `float64` precision.

## Example

```python

q = [1.0, 0.0]
K_blocks = [
    [[1.0, 0.0], [0.0, 1.0]],
    [[1.0, 1.0]]
]
V_blocks = [
    [[10.0], [20.0]],
    [[30.0]]
]

out = online_attention(q, K_blocks, V_blocks)
# approximately [18.896]
```

## What the gate checks

The gate computes a Python reference by materializing the complete attention calculation:

$$
\frac{\sum_i \exp(s_i-\max(s))V_i}
{\sum_i \exp(s_i-\max(s))}.
$$

The submitted implementation is run on multiple streamed block layouts and compared using the maximum absolute error. The result must satisfy

$$
\max |y_{\mathrm{candidate}} - y_{\mathrm{reference}}| \le 10^{-6}.
$$
