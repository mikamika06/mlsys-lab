## Context

Paged KV-cache engines (vLLM-style `PagedAttention`) store keys and values in
fixed-size physical blocks

$$
K_{\mathrm{phys}}, V_{\mathrm{phys}} \in \mathbb{R}^{B \times S \times H},
$$

where $B$ is the number of physical blocks, $S$ is the block size, and $H$ is
the head dimension. A **block table** $T \in \mathbb{N}^{L_b}$ maps logical
block position to physical block index, so the logical cache is

$$
K_{\mathrm{logical}} = \mathrm{reshape}\big([K_{\mathrm{phys}}[T_0],\, K_{\mathrm{phys}}[T_1],\, \dots],\ (-1, H)\big) .
$$

A sequence almost never ends on an exact block boundary. If the true number
of valid tokens is $n$, the last logical block only holds

$$
n - (L_b - 1)\,S
$$

valid rows; the remaining $S - \big(n - (L_b-1)S\big)$ rows in that physical
block are **stale** — leftover key/value data from whatever sequence
previously occupied that physical block (the allocator does not zero blocks
on reuse). A gather that always reads a full block of $S$ rows from every
logical block, instead of truncating to $n$, silently pulls that stale data
into attention.

## Task

Fix `gathered_attention`:

```python
def gathered_attention(k_phys, v_phys, block_table, seq_len, q):
    ...
```

- `k_phys`, `v_phys` — arrays of shape $(B, S, H)$, physical KV blocks.
- `block_table` — array of shape $(L_b,)$, physical block index per logical
  position, in logical order.
- `seq_len` — int, the true number of valid tokens $n \le L_b \cdot S$. Only
  the first `seq_len` rows of the reordered logical cache are valid; any rows
  at or beyond `seq_len` (i.e. the unused tail of the last logical block) are
  stale and **must be excluded**.
- `q` — query vector of shape $(H,)$.

Return the attention output

$$
o = \sum_i \mathrm{softmax}\!\left(\frac{q^\top K_{\mathrm{logical}}}{\sqrt{H}}\right)_{\!i}\, V_{\mathrm{logical},i},
\qquad i = 0, \dots, n-1,
$$

as a `float64` array of shape $(H,)$, computed **only** over the first `seq_len`
logical rows.

The current implementation gathers the blocks named in `block_table` and runs
attention over all $L_b \cdot S$ rows, without slicing to `seq_len`. Find and
fix the bug.

## Example

$S=4$, one logical block ($L_b=1$) built from a physical block whose first 3
rows are real tokens and whose 4th row is stale garbage from a previous
sequence. With `seq_len=3`, the correct output only attends over the first 3
rows. Including row 4 (the stale one) can shift the softmax weights — and the
output — arbitrarily, since nothing constrains what a reused, unzeroed block
happens to contain.

## What the gate checks

The gate builds physical KV blocks, injects deliberately extreme stale values
into the unused tail slots of the last logical block's backing storage, and
compares your output against a reference that gathers via `block_table` and
then truncates to `seq_len` before running attention:

$$
\max_i |o_i - \hat o_i| \le 10^{-5}.
$$

An implementation that reads a full block from every logical position —
including the partially-filled last one — will pull the stale rows into the
softmax and fail this gate by a wide margin.
