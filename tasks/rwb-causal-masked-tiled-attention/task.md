## Context

FlashAttention-style kernels never materialize the full $(n, n)$ attention
score matrix. Instead they tile the query and key/value sequences into
blocks of `block_size` rows and sweep, for each query tile, over key/value
tiles while maintaining running **online softmax** statistics — a running
row max $m$, a running row sum $\ell$, and a running (unnormalized) output
accumulator $O$ — that get rescaled every time a larger max is discovered:

$$
m_{\text{new}} = \max(m, \max_j S_{ij}), \qquad
\ell_{\text{new}} = e^{m-m_{\text{new}}}\ell + \sum_j e^{S_{ij}-m_{\text{new}}}, \qquad
O_{\text{new}} = e^{m-m_{\text{new}}}O + \sum_j e^{S_{ij}-m_{\text{new}}} V_j .
$$

After the last tile, $O / \ell$ is the final (correctly normalized)
attention output for that query tile.

For **causal** attention, row $i$ may only attend to column $j \le i$. At
tile granularity (query tile index $i_t$, key tile index $j_t$, both using
the same `block_size`) this means:

- $j_t > i_t$: the entire tile is upper-triangular (every pair masked) —
  **skip it completely**, it contributes nothing and costs nothing.
- $j_t = i_t$: the diagonal tile is only *partially* allowed — apply an
  elementwise lower-triangular mask before folding it into the running
  statistics.
- $j_t < i_t$: the entire tile is allowed — no masking needed.

## Task

Implement `tiled_causal_attention`:

```python
def tiled_causal_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, block_size: int) -> np.ndarray:
    ...
```

* `Q, K, V` — `(n, d)` arrays.
* `block_size` — tile edge length along the sequence axis (`n` need not be a
  multiple of `block_size`; the last tile is simply shorter).

Compute causal scaled dot-product attention (scale $1/\sqrt{d}$) using the
tiled/online-softmax algorithm above — skipping fully-masked key tiles,
masking only the diagonal tile, and never masking below-diagonal tiles.
Return the `(n, d)` output. The full $(n, n)$ score matrix must never be
computed at once — only tile-sized score blocks $(≤\text{block\_size}) \times (≤\text{block\_size})$.

## Example

```python
import numpy as np

n, d, block_size = 7, 4, 3
Q = np.random.randn(n, d)
K = np.random.randn(n, d)
V = np.random.randn(n, d)

out = tiled_causal_attention(Q, K, V, block_size)
# out.shape == (7, 4); mathematically identical to standard causal
# softmax(QK^T / sqrt(d)) @ V, just computed tile-by-tile.
```

## What the gate checks

A single gate, **max_abs_err**, compares your output against a direct
(untiled) causal softmax-attention oracle computed in `float64`, across 8
random trials with random `n`, `d`, and `block_size` (including cases where
`block_size` does not evenly divide `n`, exercising the ragged tail tile).
Must be `<= 1e-4`.
