## Context

A KV-cache quantizer keeps the **most recent** $R$ tokens of the Key/Value cache in full
precision (they matter most for attention) and quantizes the remaining, older
$T - R$ tokens with per-row affine (min–max) quantization, where $T$ is the
sequence length. The two caches can even use **different bit-widths**:
$\text{nbits}_K$ for keys and $\text{nbits}_V$ for values.

Given a byte budget $B$, a production serving stack must pick, among a small
set of candidate configurations $(\text{nbits}_K, \text{nbits}_V, R)$, the one
that **minimizes total reconstruction MSE** while fitting inside $B$ bytes.

### Per-row affine quantization

For a matrix $X \in \mathbb{R}^{n \times d}$ split into groups of size
`group_size` along the last axis, each group $g$ (with values $x_1,\dots,x_m$)
is quantized with $b$ bits as:

$$
\text{scale} = \frac{\max(g) - \min(g)}{2^{b}-1}, \qquad
\text{code}_i = \mathrm{clip}\left(\mathrm{round}\!\left(\frac{x_i - \min(g)}{\text{scale}}\right),\,0,\,2^b-1\right)
$$

$$
\hat{x}_i = \text{code}_i \cdot \text{scale} + \min(g)
$$

(if `scale == 0`, treat it as `1.0` to avoid division by zero — reconstruction
is then exact). This is *symmetric range, asymmetric (min-based) affine*
quantization — a common scheme (e.g. KIVI-style KV quantizers).

### Cost of one configuration

For a config $(\text{nbits}_K, \text{nbits}_V, R)$ applied to keys $K$ and
values $V$, both of shape $(T, d)$:

* The **oldest** $T-R$ rows of $K$ are quantized with $\text{nbits}_K$ bits
  (per-row groups of size `group_size`); the **oldest** $T-R$ rows of $V$ are
  quantized with $\text{nbits}_V$ bits. The most recent $R$ rows of both $K$
  and $V$ are kept exact (float32, zero error).
* **MSE** = sum of squared reconstruction error over all quantized elements of
  $K$ and $V$, divided by the total element count of $K$ and $V$ combined
  (window elements contribute `0` error but still count in the denominator).
* **Bytes** = for each tensor's quantized part: `ceil(n_rows * n_groups *
  group_size * nbits / 8)` code bytes, plus `n_rows * n_groups * 2 * 4`
  overhead bytes for one float32 `scale` and one float32 `min` per group per
  row. Plus, for the window part of each tensor: `n_window_rows * d * 4`
  bytes (stored as float32). Total bytes = sum over $K$ and $V$.

A config is **feasible** if its total bytes $\le B$.

## Task

Implement `choose_kv_budget`:

```python
def choose_kv_budget(
    K: np.ndarray,
    V: np.ndarray,
    candidates: list[tuple[int, int, int]],
    byte_budget: int,
    group_size: int,
) -> int:
    ...
```

* `K`, `V` — float arrays of shape `(T, d)` (`d` is a multiple of `group_size`).
* `candidates` — a list of `(nbits_K, nbits_V, R)` triples to consider, in a
  fixed order.
* `byte_budget` — integer byte budget `B`.
* `group_size` — quantization group size along the feature axis.

Return the **index into `candidates`** of the feasible configuration (total
bytes $\le B$) with the **smallest MSE**, exactly as defined above. `byte_budget`
is guaranteed to admit at least one feasible candidate. Ties (equal MSE) are
broken by picking the smaller index — i.e. plain `argmin` semantics over the
feasible subset in candidate order.

## Example

```python
import numpy as np
K = np.random.default_rng(0).normal(size=(16, 4))
V = np.random.default_rng(1).normal(size=(16, 4))
candidates = [(2, 2, 0), (4, 4, 0), (2, 4, 4), (4, 2, 8)]
idx = choose_kv_budget(K, V, candidates, byte_budget=400, group_size=4)
# idx is the position in `candidates` of the cheapest-MSE config that fits in 400 bytes
```

## What the gate checks

Gate **argmin_index** re-implements the exact cost/MSE formulas above as a
NumPy oracle, enumerates the same `candidates` list on several random
`(K, V, byte_budget)` instances, and compares the index your function returns
against the oracle's `argmin` over the feasible subset. Any mismatch fails the
gate — there is no numeric tolerance on the index itself.
