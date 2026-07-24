## Context

Converting a pretrained Multi-Head Attention (MHA) checkpoint into a
Grouped-Query Attention (GQA) checkpoint requires collapsing the $H$
independent key/value heads down to $G$ shared key/value heads (with
$H$ divisible by $G$), while keeping all $H$ query heads. A common,
cheap initialization for the new GQA weights — used before any further
fine-tuning — is to **mean-pool** each contiguous group of
$r = H / G$ original KV heads into one KV head.

Let $K, V \in \mathbb{R}^{B \times H \times T \times D}$ be the batched
MHA key/value tensors (batch, heads, sequence, head-dim). Partition the
$H$ heads into $G$ contiguous groups of size $r = H/G$:

$$
\text{group}(g) = \{\, g\cdot r,\ g\cdot r + 1,\ \dots,\ g\cdot r + r - 1 \,\}, \qquad g = 0,\dots,G-1 .
$$

The pooled GQA tensors $K', V' \in \mathbb{R}^{B \times G \times T \times D}$ are

$$
K'[b,g,t,:] = \frac{1}{r}\sum_{h \,\in\, \text{group}(g)} K[b,h,t,:],
\qquad
V'[b,g,t,:] = \frac{1}{r}\sum_{h \,\in\, \text{group}(g)} V[b,h,t,:].
$$

Each of the $r$ query heads that mapped to original KV head $h$ now
attends against the *shared* pooled head $K'[b,g]$, $V'[b,g]$ instead of
its own private $K[b,h]$, $V[b,h]$.

## Task

Implement `mha_to_gqa_pool`:

```python
def mha_to_gqa_pool(K: np.ndarray, V: np.ndarray, n_kv_heads: int) -> tuple[np.ndarray, np.ndarray]:
    ...
```

* `K`, `V` — `float64` arrays of shape $(B, H, T, D)$: the original MHA
  key and value tensors (one KV head per query head).
* `n_kv_heads` — the target number of GQA key/value heads $G$. $H$ is
  guaranteed divisible by $G$.

Group the $H$ heads into $G$ **contiguous** blocks of size $r = H/G$ (in
head-index order — heads $0,\dots,r-1$ form group $0$, heads
$r,\dots,2r-1$ form group $1$, etc.) and mean-pool $K$ and $V$ within
each block as defined above.

Return `(K_gqa, V_gqa)`, each of shape $(B, G, T, D)$.

## Example

```python
import numpy as np

B, H, T, D, G = 1, 4, 2, 3, 2
K = np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D)
V = K + 100.0

K_gqa, V_gqa = mha_to_gqa_pool(K, V, G)
print(K_gqa.shape)   # (1, 2, 2, 3)
# K_gqa[0, 0] == mean(K[0, 0], K[0, 1])   (heads 0,1 pooled into group 0)
# K_gqa[0, 1] == mean(K[0, 2], K[0, 3])   (heads 2,3 pooled into group 1)
```

## What the gate checks

The grader builds several deterministic `(K, V, n_kv_heads)` cases,
computes the reference pooled tensors with the reshape-and-mean formula
above, and treats each of the $G$ output KV heads as one "channel".
**channel_rel_err** is the *mean, over the $G$ channels*, of the
per-channel relative $L_2$ reconstruction error between your pooled
$(K', V')$ and the reference — so a poorly reconstructed head counts just
as much as a well-reconstructed one, regardless of magnitude. It must be
$\le 10^{-6}$.
