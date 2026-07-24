## Context

The simplest slice of FlashAttention's tiling scheme keeps the whole query
block fixed in registers/cache and streams the key/value matrix past it in
tiles — no tiling of $Q$ at all, just $K,V$. For a fixed query tile
$Q \in \mathbb{R}^{n_q \times d}$ and key/value tiles $K_j, V_j$ of size
$\text{kv\_block\_size}$ each, the online-softmax recurrence maintains a
per-query running max $m$, running normalizer $\ell$, and running (unnormalized)
output accumulator $O$:

$$
S_j = \frac{Q K_j^\top}{\sqrt d}, \qquad
m_{\text{new}} = \max(m, \mathrm{rowmax}(S_j)),
$$
$$
P_j = e^{S_j - m_{\text{new}}}, \qquad
\ell \leftarrow \ell \cdot e^{m - m_{\text{new}}} + \mathrm{rowsum}(P_j),
$$
$$
O \leftarrow O \cdot e^{m - m_{\text{new}}} + P_j V_j, \qquad
m \leftarrow m_{\text{new}} .
$$

Starting from $m=-\infty$, $\ell=0$, $O=0$ and sweeping every KV tile in
order, the final $O / \ell$ is *exactly* the dense softmax-attention output
$\mathrm{softmax}(QK^\top/\sqrt d)\,V$ — the rescale-by-$e^{m-m_{\text{new}}}$
step is what keeps a running softmax mathematically identical to the
softmax over everything seen so far, no matter how the keys are chunked.

## Task

Implement `flash_forward_single_q_tile`:

```python
def flash_forward_single_q_tile(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                 kv_block_size: int) -> np.ndarray:
    ...
```

* `Q` — float array of shape $(n_q, d)$: the single, fixed query tile (do
  **not** sub-tile $Q$ — process all of it as one block).
* `K`, `V` — float arrays of shape $(n_{kv}, d)$: the full key/value
  matrices, to be streamed in tiles of `kv_block_size` rows each (the last
  tile may be shorter).
* Returns a `float64` array of shape $(n_q, d)$.

You must actually iterate over KV tiles of size `kv_block_size`, updating
the running $(m, \ell, O)$ triple after each tile with the recurrence
above — not compute the dense $(n_q, n_{kv})$ score matrix in one shot and
ignore `kv_block_size`.

## Example

```python
import numpy as np
rng = np.random.default_rng(0)
n_q, n_kv, d = 8, 100, 4
Q = rng.standard_normal((n_q, d))
K = rng.standard_normal((n_kv, d))
V = rng.standard_normal((n_kv, d))
out = flash_forward_single_q_tile(Q, K, V, kv_block_size=16)
# out.shape == (8, 4), matches dense softmax(Q @ K.T / sqrt(d)) @ V within 1e-5,
# regardless of kv_block_size (16, 25, 100, ...) used to stream K, V.
```

## What the gate checks

* **max_abs_err** — your output vs. a dense float64 NumPy oracle
  `softmax(Q @ K.T / sqrt(d)) @ V`, element-wise, over several seeded
  random `(Q, K, V, kv_block_size)` cases, must be $\le 10^{-5}$.
* **loop_ratio** — the grader runs your function twice on the same
  `(Q, K, V)` with `kv_block_size` set to (a) the full $n_{kv}$ (one KV
  tile) and (b) $n_{kv}/8$ (eight KV tiles), counting Python-level line
  executions with a tracer both times (after a warm-up call). Genuinely
  looping over KV tiles makes the 8-tile run execute markedly more Python
  lines than the 1-tile run; the ratio must be $\ge 2.0$. A solution that
  ignores `kv_block_size` and always computes the dense attention in one
  shot passes **max_abs_err** but produces a ratio of about `1.0` and fails
  this gate.
