## Context

PyTorch's `flex_attention` generalizes attention by letting a user supply a
`score_mod(score, b, h, q_idx, kv_idx)` function that rewrites every
attention logit before the softmax, given its batch index `b`, head index
`h`, query position `q_idx`, and key/value position `kv_idx`.

ALiBi (Attention with Linear Biases) is one such score modification. Instead
of adding position information to $Q$ or $K$ directly, it adds a bias to the
raw attention score that grows linearly with the signed distance between the
query and key positions, scaled by a per-head slope $m_h$:

$$
\mathrm{score\_mod}(S_{h,ij}, h, i, j) = S_{h,ij} + m_h \cdot (j - i)
$$

where $i$ is the query index (`q_idx`), $j$ is the key index (`kv_idx`), and
$S_h = \dfrac{Q_h K_h^\top}{\sqrt{d}}$ is the scaled dot-product score matrix
for head $h$.

The full attention computation for head $h$ is then

$$
\mathrm{Attention}_h(Q,K,V) = \mathrm{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d}} + m_h \cdot (j - i)\right) V_h,
$$

with the row-wise softmax

$$
\mathrm{softmax}(x)_j = \frac{e^{x_j - \max_k x_k}}{\sum_k e^{x_k - \max_k x_k}}.
$$

Different heads use different slopes $m_h$, letting some heads attend almost
only to nearby positions while others stay closer to uniform attention.

## Task

Implement `alibi_score_mod_attention(Q, K, V, slopes)`:

```python
def alibi_score_mod_attention(Q, K, V, slopes):
    ...
```

Inputs are NumPy arrays:
- `Q` has shape $(H, n, d)$ — one query block per head.
- `K` has shape $(H, m, d)$.
- `V` has shape $(H, m, d_v)$.
- `slopes` has shape $(H,)$ — one positive ALiBi slope $m_h$ per head.

For every head $h$, query position $i$ (`0 <= i < n`), and key position $j$
(`0 <= j < m`), apply the score modification

$$
S'_{h,ij} = \frac{(Q_h K_h^\top)_{ij}}{\sqrt{d}} + m_h \, (j - i)
$$

before the softmax. Return the attention output as a `float64` NumPy array
of shape $(H, n, d_v)$:

$$
\mathrm{out}_h = \mathrm{softmax}(S'_h)\, V_h .
$$

This is unmasked (bidirectional) attention — every query position attends to
every key position, just with the linear ALiBi penalty/bonus applied to the
logit first. Vectorize across heads and positions; avoid Python-level loops
over `q_idx`/`kv_idx` pairs.

## Example

```python
import numpy as np

Q = np.array([[[1.0, 0.0], [0.0, 1.0]]])   # H=1, n=2, d=2
K = np.array([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])  # H=1, m=3, d=2
V = np.array([[[1.0], [2.0], [3.0]]])      # H=1, m=3, d_v=1
slopes = np.array([0.5])

out = alibi_score_mod_attention(Q, K, V, slopes)
# out.shape == (1, 2, 1)
```

For query index `i=0` and key index `j=2`, the raw score
`(Q[0] @ K[0].T)[0, 2] / sqrt(2)` gets `0.5 * (2 - 0) = 1.0` added to it
before the softmax over `j`.

## What the gate checks

The gate builds several random `(H, n, m, d, d_v)` cases with
`np.random.default_rng(0)` and computes, in `float64`, the same scaled
dot-product scores, adds the exact ALiBi bias `slopes[h] * (kv_idx - q_idx)`
per head via a position-difference grid, applies a numerically-stable
row-wise softmax, and contracts with `V`. It compares this oracle against
`alibi_score_mod_attention`'s output with the maximum absolute error metric;
the error must stay below $10^{-5}$ on every case.
