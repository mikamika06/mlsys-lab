## Context

Standard multi-head attention (MHA) computes $n_h$ independent attention heads,
each with its own query, key, and value projections. Grouped-query attention
(GQA) reduces the key-value cache by using only $n_{kv} < n_h$ KV heads: each
KV head is shared across a contiguous group of $q = n_h / n_{kv}$ query heads.
Multi-query attention (MQA) is the extreme case $n_{kv} = 1$.

Given tensors with the following shapes:

- $Q \in \mathbb{R}^{B \times T_q \times n_h \times d_h}$ (queries)
- $K \in \mathbb{R}^{B \times T_k \times n_{kv} \times d_h}$ (keys)
- $V \in \mathbb{R}^{B \times T_k \times n_{kv} \times d_h}$ (values)

the GQA forward pass first expands $K$ and $V$ so each KV head covers its full
query group:

$$\tilde{K}[b,\, t,\, j q : (j{+}1) q,\, :] \;=\; K[b,\, t,\, j,\, :], \qquad j = 0, \dots, n_{kv} - 1,$$

and analogously for $\tilde{V}$. Then standard scaled dot-product attention runs
on the expanded tensors:

$$\text{scores} = \frac{Q\, \tilde{K}^{\!\top}}{\sqrt{d_h}}, \qquad
  W = \text{softmax}(\text{scores}), \qquad
  O = W\, \tilde{V},$$

where softmax is applied along the key-sequence dimension. The result is
transposed back to $(B, T_q, n_h, d_h)$.

## Task

Implement `gqa_attention(Q, K, V)`:

```python
import numpy as np

def gqa_attention(Q, K, V):
    """Compute grouped-query attention.

    Args:
        Q: np.ndarray, shape (batch, seq_q, n_heads, head_dim)
        K: np.ndarray, shape (batch, seq_k, n_kv_heads, head_dim)
        V: np.ndarray, shape (batch, seq_k, n_kv_heads, head_dim)

    Returns:
        np.ndarray, shape (batch, seq_q, n_heads, head_dim), dtype float64
    """
```

You are guaranteed `n_heads % n_kv_heads == 0`. Use NumPy only (no PyTorch,
no JAX). Include numerically stable softmax.

## Example

```python
import numpy as np
rng = np.random.RandomState(0)
B, T, n_h, n_kv, d = 1, 4, 4, 2, 8
Q = rng.randn(B, T, n_h, d)
K = rng.randn(B, T, n_kv, d)
V = rng.randn(B, T, n_kv, d)
out = gqa_attention(Q, K, V)
assert out.shape == (B, T, n_h, d)   # (1, 4, 4, 8)
```

## What the gate checks

The metric `max_abs_err` is the worst-case element-wise absolute difference
against a NumPy oracle that expands KV heads with `np.repeat` and then runs
standard scaled dot-product attention with numerically stable softmax. Five
configurations are tested: GQA with group sizes 4 and 2, MQA ($n_{kv} = 1$),
MHA ($n_{kv} = n_h$), and cross-attention where $T_q \ne T_k$. The gate passes
when `max_abs_err < 1e-6`.
