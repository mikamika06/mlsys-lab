## Context

In grouped-query attention (GQA) each of $H_q$ query heads is paired with one
of $H_{kv} \le H_q$ key/value heads, $H_q = H_{kv}\cdot g$ for group size $g$;
query head $h$ shares KV head $\lfloor h/g \rfloor$. For a single decode step
with query $q_h \in \mathbb{R}^{D}$ and that head's key/value sequence
$K_h, V_h \in \mathbb{R}^{L \times D}$,

$$
o_h = \sum_i \mathrm{softmax}\!\left(\frac{q_h^\top K_h}{\sqrt D}\right)_{\!i} V_{h,i}.
$$

A paged KV-cache engine never stores $K,V$ contiguously. Instead it keeps
fixed-size physical blocks

$$
K_{\mathrm{phys}}, V_{\mathrm{phys}} \in \mathbb{R}^{B \times S \times H_{kv} \times D},
$$

where $B$ is the number of physical blocks and $S$ is tokens per block —
every physical block holds all $H_{kv}$ heads for $S$ tokens at once, and
physical blocks for one sequence can land anywhere, in any order, mixed in
with blocks nobody is using. A **block table** $T \in \mathbb{N}^{L_b}$
($L_b = L/S$) records, per logical block position, which physical block
actually holds that data:

$$
K_{\mathrm{logical}} = \mathrm{reshape}\big([K_{\mathrm{phys}}[T_0], \dots, K_{\mathrm{phys}}[T_{L_b-1}]],\ (L, H_{kv}, D)\big).
$$

The claim this task verifies: attention computed by gathering through
$T$ and running GQA on the reconstructed logical sequence is **numerically
identical** to attention computed directly on the original contiguous
sequence — the physical layout is pure indirection, invisible to the math.

## Task

Implement `paged_gqa_attention`:

```python
def paged_gqa_attention(k_phys, v_phys, block_table, q, n_kv_heads):
    ...
```

- `k_phys`, `v_phys` — arrays of shape $(B, S, H_{kv}, D)$: physical KV
  blocks, all heads interleaved per token, per the layout above. Physical
  blocks not referenced by `block_table` may hold unrelated leftover data.
- `block_table` — array of shape $(L_b,)$: physical block index per logical
  block position, in logical order.
- `q` — array of shape $(H_q, D)$: one query vector per query head, for a
  single decode step. $H_q$ is a multiple of `n_kv_heads`.
- `n_kv_heads` — int, $H_{kv}$.

Gather the logical KV sequence via `block_table`, then compute GQA attention
(query head $h$ against KV head $\lfloor h \cdot H_{kv} / H_q \rfloor$,
i.e. contiguous grouping) and return a `float64` array of shape $(H_q, D)$.

## Example

With $S=2$, one logical block ($L_b=1$) whose 2 tokens actually live in
physical block `3`, `block_table = [3]` means
`k_logical = k_phys[3]` — the physical positions before or after block `3`
are irrelevant no matter what garbage they contain.

## What the gate checks

The grader loads one fixed GQA layer's $K$, $V$, $Q$ (128 tokens, 8 KV
heads, group size 4, head dim 64), scatters the tokens into a larger pool of
physical blocks at a random permutation of positions (extra, unused physical
slots are filled with distinct garbage values), and builds the matching
`block_table` — across three different block sizes. It compares your output
against attention computed directly on the original contiguous $K,V,Q$:

$$
\max_i |o_i - \hat o_i| < 10^{-6}.
$$

Indexing into the wrong physical block, mixing up which KV head a query head
should attend to, or including any of the deliberately garbage-filled unused
physical blocks will produce a large, easily detected deviation.
