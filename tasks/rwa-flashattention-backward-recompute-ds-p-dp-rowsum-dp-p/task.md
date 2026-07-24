## Context

FlashAttention avoids materializing the full attention matrix during the backward pass. The forward pass stores the row-wise softmax statistics needed to reconstruct the probabilities later.

For queries $Q \in \mathbb{R}^{n \times d}$ and keys $K \in \mathbb{R}^{n \times d}$, attention scores are

$$
S = QK^\top .
$$

The saved statistics are the row maximums $m$ and row normalizers $l$:

$$
m_i = \max_j S_{ij},
\qquad
l_i = \sum_j \exp(S_{ij} - m_i).
$$

The attention probabilities can be recomputed without storing the full matrix:

$$
P_{ij} = \frac{\exp(S_{ij} - m_i)}{l_i}.
$$

Given the output gradient $dO$, the attention backward pass computes

$$
dP = dO V^\top .
$$

The softmax backward identity used by FlashAttention is

$$
dS = P \circ \left(dP - \sum_j dP_{ij}P_{ij}\right),
$$

where the summation is performed independently for every row. The remaining gradients are

$$
dQ = dS K,
$$

$$
dK = dS^\top Q,
$$

$$
dV = P^\top dO.
$$

## Task

Implement `flash_attention_backward(q, k, v, do, m, l)`.

The function receives NumPy arrays:
- `q` with shape $(n,d)$
- `k` with shape $(n,d)$
- `v` with shape $(n,d)$
- `do` with shape $(n,d)$
- `m` with shape $(n,1)$ containing saved row maxima
- `l` with shape $(n,1)$ containing saved row softmax normalizers

It must return a tuple `(dq, dk, dv)` containing the gradients with respect to $Q$, $K$, and $V$. Recompute $P$ from `q`, `k`, `m`, and `l`; do not use a stored attention matrix.

The outputs must be `float64` NumPy arrays.

## Example

```python
import numpy as np

q = np.array([[1.0, 0.0], [0.0, 1.0]])
k = np.array([[1.0, 0.0], [0.0, 1.0]])
v = np.array([[2.0, 0.0], [0.0, 3.0]])
do = np.ones((2, 2))

s = q @ k.T
m = np.max(s, axis=1, keepdims=True)
l = np.sum(np.exp(s - m), axis=1, keepdims=True)

dq, dk, dv = flash_attention_backward(q, k, v, do, m, l)
```

## What the gate checks

The gate builds a numerical oracle by applying central finite differences to the naive attention computation. It compares the returned $dQ$, $dK$, and $dV$ against the oracle gradients using the maximum absolute error.

The maximum absolute error

$$
\max_i |x_i-\hat{x}_i|
$$

over all three returned gradient tensors must be at most $10^{-4}$. Implementations that use an incorrect softmax row sum or fail to recompute probabilities will not pass.
