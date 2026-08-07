## Context

Standard scaled dot-product attention over queries $Q$, keys $K$, values $V \in \mathbb{R}^{N \times d}$ is

$$
\text{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt d}\right) V .
$$

Computed directly, this materializes the full $N \times N$ score matrix
$S = QK^\top/\sqrt d$. For long sequences that matrix dominates memory —
$O(N^2)$ against an output that is only $O(Nd)$.

FlashAttention-2 avoids ever forming $S$ in full. It tiles $Q$ into row
blocks and $K,V$ into column blocks, and for each query block sweeps over
key/value blocks while maintaining, per query row, a running max $m$ and
running normalizer $\ell$ (the **online softmax**). Whenever a new block
raises the running max, the accumulated output and normalizer are rescaled
by $e^{m_{\text{old}} - m_{\text{new}}}$ before adding the new block's
contribution — this is exactly what keeps the running softmax numerically
equal to the softmax over all keys seen so far, without ever exponentiating
against a stale, too-small max.

For query block $Q_i$ against key/value block $K_j, V_j$:

$$
S_{ij} = \frac{Q_i K_j^\top}{\sqrt d}, \qquad
m_{\text{new}} = \max(m, \mathrm{rowmax}(S_{ij})),
$$
$$
P_{ij} = e^{S_{ij} - m_{\text{new}}}, \qquad
\ell \leftarrow \ell \cdot e^{m - m_{\text{new}}} + \mathrm{rowsum}(P_{ij}),
$$
$$
O_i \leftarrow O_i \cdot e^{m - m_{\text{new}}} + P_{ij} V_j, \qquad
m \leftarrow m_{\text{new}} .
$$

After the last key/value block, $O_i \leftarrow O_i / \ell$ gives the exact
attention output for that query block — identical to the dense computation,
up to floating-point rounding, but with peak memory bounded by the block
size instead of $N$.

## Task

Implement `flash_attention_forward`:

```python
def flash_attention_forward(Q: list[list[float]], K: list[list[float]], V: list[list[float]], block_size: int=32) -> list[list[float]]:
    ...
```

- `Q`, `K`, `V` — list of lists of floats of shape $(N, d)$.
- `block_size` — tile size used for both the query and the key/value
  dimension.
- Returns a `float64` array of shape $(N, d)$, the attention output.

Your implementation must sweep `Q`, `K`, `V` in tiles of at most
`block_size` rows using the running-max / running-sum recurrence above.
**At no point may an $(N, N)$ (or larger) array be allocated** — only
block-sized intermediates (at most `block_size` $\times$ `block_size`, or
`block_size` $\times$ `d`) and the $(N, d)$ output/accumulator are allowed.

## Example

```python
rng = random.Random(0)
N, d = 64, 8
Q = rng.standard_normal((N, d))
K = rng.standard_normal((N, d))
V = rng.standard_normal((N, d))
out = flash_attention_forward(Q, K, V, block_size=16)
# out shape (64, 8), matches dense softmax(Q @ K.T / sqrt(d)) @ V within 1e-4
```

## What the gate checks

Two gates, both against sequences long enough that an $N \times N$ buffer
would dwarf a properly tiled one:

- **max_abs_err** — your output vs. a dense float64 Python oracle
  `softmax(Q @ K.T / sqrt(d)) @ V`, element-wise, must be $\le 10^{-4}$.
- **peak_alloc_ratio** — the grader measures real peak traced memory
  (via `tracemalloc`, after a warm-up call so one-time allocator setup
  doesn't pollute the measurement) during your call, and divides it by the
  byte size of a single float64 $N \times N$ matrix. A solution that forms
  the full score matrix even once uses several times that many bytes
  (the matrix itself plus its softmax intermediates); a properly tiled
  solution uses a small fraction of it. The gate requires the ratio to be
  $\le 0.5$.

An implementation that is numerically correct but computes `Q @ K.T` in one
shot will pass the first gate and fail the second.
