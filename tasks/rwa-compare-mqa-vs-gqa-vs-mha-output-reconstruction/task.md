## Context

Multi-head attention (MHA) gives every query head $h$ its own key/value pair
$(K_h, V_h)$. That is expensive to cache during autoregressive decoding: the
KV cache grows linearly in the number of heads. Two common ways to shrink it:

- **GQA($g$)** (grouped-query attention): partition the $n_h$ heads into
  groups of $g$ adjacent heads, and store **one shared KV pair per group**
  (typically the group's KV pair is the original K/V averaged, or one of
  them chosen as a representative — here we use the mean). Every query head
  in the group attends against its group's shared KV.
- **MQA** (multi-query attention): the extreme case $g = n_h$ — a single
  shared KV pair for *all* heads.
- **MHA** is the trivial case $g = 1$ — no sharing at all.

Given the *same* $Q, K, V$ (as if $K, V$ were originally computed per-head,
exactly like MHA), reconstructing the output under GQA($g$)/MQA means:
mean-pool $K$ and $V$ within each group of $g$ heads, broadcast the pooled
KV back out to all $n_h$ heads, then run ordinary scaled dot-product
attention with the *original* $Q$:

$$
\bar K_{:,j} = \frac{1}{g}\sum_{h \in \text{group}(j)} K_{:,h}, \qquad
\text{score}_h = \frac{Q_h \bar K_{\text{group}(h)}^\top}{\sqrt{d_k}}, \qquad
\text{out}_h = \operatorname{softmax}(\text{score}_h)\, \bar V_{\text{group}(h)}
$$

Coarser grouping (larger $g$) throws away more per-head information before
attention even runs, so the reconstructed output should drift further from
the true MHA output as $g$ grows — while the KV cache shrinks in exact
proportion, by a factor of $g$.

## Task

Implement `mha_gqa_mqa_reconstruct(Q, K, V, group_sizes)`:

```python
def mha_gqa_mqa_reconstruct(Q: list[list[list[list[float]]]], K: list[list[list[list[float]]]], V: list[list[list[list[float]]]], group_sizes):
    ...
```

- `Q`: shape `(batch, seq_q, n_heads, head_dim)`.
- `K`, `V`: shape `(batch, seq_k, n_heads, head_dim)` — one KV pair per
  head, as in ordinary MHA.
- `group_sizes`: an iterable of positive ints, each evenly dividing
  `n_heads`. `1` means MHA (no pooling), `n_heads` means MQA, anything else
  is GQA($g$).

For **every** `g` in `group_sizes`, in order:

1. Reshape the head axis of `K`/`V` into `(n_heads // g, g)` groups of `g`
   adjacent heads and mean-pool over the group axis.
2. Broadcast (repeat) each group's pooled K/V back out to all `g` heads in
   that group, restoring shape `(batch, seq_k, n_heads, head_dim)`.
3. Run standard scaled dot-product attention: `Q` against the
   broadcast K, softmax with scale `1/sqrt(head_dim)`, weighted sum of the
   broadcast V.
4. Compute `size_ratio = (n_heads // g) / n_heads` — the grouped KV cache
   size relative to storing one KV pair per head.

Return a list, one `(output, size_ratio)` tuple per entry of `group_sizes`,
in the same order:

- `output`: `ndarray` of shape `(batch, seq_q, n_heads, head_dim)`.
- `size_ratio`: a Python float.

No explicit Python loop over heads/batch/sequence positions is needed — use
vectorised Python (`reshape`, `mean`, `repeat`, batched matmul).

## Example

```python

rng = random.Random(0)
Q = rng.standard_normal((1, 3, 4, 8))
K = rng.standard_normal((1, 3, 4, 8))
V = rng.standard_normal((1, 3, 4, 8))

results = mha_gqa_mqa_reconstruct(Q, K, V, [1, 2, 4])
for g, (out, ratio) in zip([1, 2, 4], results):
    print(g, out.shape, ratio)
# 1 (1, 3, 4, 8) 1.0     <- MHA, no pooling
# 2 (1, 3, 4, 8) 0.5     <- GQA(2), 2 KV heads instead of 4
# 4 (1, 3, 4, 8) 0.25    <- MQA, 1 shared KV head
```

## What the gate checks

The oracle builds several `(Q, K, V)` cases from a seeded generator across
different batch sizes, sequence lengths, head counts, and head dims, each
paired with a list of `group_sizes` that includes `1` (exact MHA), one or
more intermediate GQA arities, and the full MQA case (`g == n_heads`). For
every case and every `g`, it independently mean-pools, broadcasts, and runs
attention exactly as described above.

Your returned `output` tensors are compared element-for-element against the
oracle's outputs (`max_abs_err`, threshold `1e-6`) for **every** `g` in
every case — so the `g == 1` entry must reproduce plain MHA exactly (no
accidental pooling), and every other entry must reproduce the correct
mean-pool-then-broadcast reconstruction. Your `size_ratio` values are
checked for exact equality against `(n_heads // g) / n_heads`. Missing the
`1/sqrt(head_dim)` scale, pooling across the wrong axis, broadcasting
incorrectly (e.g. reversing which heads share which pooled KV), or
returning a size ratio computed the wrong way round (e.g. `g / n_heads`
instead of `(n_heads // g) / n_heads`) will fail the gate.
