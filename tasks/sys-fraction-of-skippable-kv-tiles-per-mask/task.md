## Context

A tiled attention kernel partitions the $n \times n$ attention mask into
$\text{tile\_size} \times \text{tile\_size}$ blocks — one block per
(query-tile, KV-tile) pair. If a block is **entirely** `False` (no query
in that query-tile attends to any key in that KV-tile), a blocked kernel
never needs to compute scores for it at all: the block can be skipped.

This holds for every masking scheme, not just causal masking:

- **Causal**: $M_{ij} = [\,j \le i\,]$ — blocks strictly above the
  diagonal are fully `False`.
- **Sliding window** (width $w$): $M_{ij} = [\,j \le i\,] \wedge [\,j > i - w\,]$
  — blocks fully above the diagonal *and* blocks fully more than $w$
  positions before the diagonal are fully `False`.
- **Block-sparse**: an arbitrary pattern, sometimes already `False` for
  whole blocks by construction.

The fraction of *fully-`False`* blocks (over all $\lceil n/\text{tile\_size}\rceil^2$
blocks) tells you how much compute a correctly tiled kernel can skip for a
given mask and tile size:

$$
\text{skippable\_fraction} = 1 - \frac{1}{n_t^2}\sum_{q=0}^{n_t-1}\sum_{k=0}^{n_t-1}
\mathbb{1}\!\left[\exists\, (i,j) \in \text{block}(q,k) : M_{ij} = \text{True}\right]
$$

where $n_t = n / \text{tile\_size}$.

## Task

Implement `skippable_kv_tile_fraction`:

```python
def skippable_kv_tile_fraction(mask: list[list[bool]], tile_size: int) -> float:
    ...
```

- `mask`: `(n, n)` boolean list. `True` means "this query attends
  to this key" (kept); `False` means masked out. `n` is guaranteed to be
  divisible by `tile_size`.
- `tile_size`: side length of the square (query-tile, KV-tile) blocks.

Partition `mask` into a `(n // tile_size) x (n // tile_size)` grid of
`tile_size x tile_size` blocks. Return the fraction of those blocks that
are entirely `False` (i.e. every element inside them is `False`) as a
plain Python `float` in `[0.0, 1.0]`.

## Example

```python

n, tile_size = 4, 2
mask = [
    [True,  False, False, False],
    [True,  True,  False, False],
    [True,  True,  True,  False],
    [True,  True,  True,  True],
]  # causal mask

frac = skippable_kv_tile_fraction(mask, tile_size)
# blocks: (0,0)=[[T,F],[T,T]] has a True -> kept
#         (0,1)=[[F,F],[F,F]] all False  -> skippable
#         (1,0)=[[T,T],[T,T]] has a True -> kept
#         (1,1)=[[T,F],[T,T]] has a True -> kept
# 1 of 4 blocks is fully skippable -> frac == 0.25
```

## What the gate checks

The grader builds several masks with a real Python oracle — causal,
sliding-window, a random block-sparse pattern (generated at tile
granularity so its true fraction is known exactly), a random
element-wise-sparse mask (so the block reduction must be genuine, not a
tile-level shortcut), an all-masked mask, and an all-kept mask — each
with its own tile size. For every case it independently computes the true
skippable-block fraction by reshaping the mask into blocks and checking
`.any()` per block.

`size_ratio` is the worst-case absolute difference between your returned
fraction and the true fraction across all cases (must be `< 1e-9`) — this
requires an exact block reduction, not an approximation, and the random
element-wise mask specifically rules out any solution that only handles
uniform block patterns.
