## Context

Paged-attention implementations keep each sequence's key-value (KV) cache
as a chain of fixed-size *blocks*, allocated from a pool the way an OS
allocates physical pages. Each token's KV entry is `s` bytes; the
allocator rounds that up to a whole number of `b`-byte blocks:

$$
\text{blocks}(s, b) = \left\lceil \frac{s}{b} \right\rceil, \qquad
\text{bytes}(s, b) = \text{blocks}(s, b) \cdot b .
$$

Rounding up wastes up to $b-1$ bytes per token (*internal
fragmentation*) — worse for larger $b$. But every block also costs
`table_overhead_per_block` bytes of block-table metadata (an entry
identifying which physical block backs it), and a larger $b$ means
*fewer* blocks per token, so *less* total table overhead. Block size
trades one kind of waste for the other.

## Task

Implement

```cpp
void choose_kv_block_size(const int* token_sizes, int n, int table_overhead_per_block,
                           int* out_block_size, long* out_useful_bytes, long* out_allocated_bytes);
```

Candidate block sizes are the powers of two `{16, 32, 64, 128, 256, 512,
1024}`. For a workload of `n` tokens `token_sizes[0..n)`, and a candidate
`b`:

$$
\text{allocated}(b) = \sum_i \text{bytes}(\text{token\_sizes}[i], b)
                       + \text{total\_blocks}(b) \cdot \text{table\_overhead\_per\_block}
$$

where $\text{total\_blocks}(b) = \sum_i \text{blocks}(\text{token\_sizes}[i], b)$.
The useful bytes, $\sum_i \text{token\_sizes}[i]$, is the **same for every
candidate** — so the block size maximizing $\text{useful}/\text{allocated}(b)$
is exactly the one **minimizing** $\text{allocated}(b)$, an integer
comparison with no floating point involved.

Write the chosen block size to `*out_block_size`, the useful-byte total to
`*out_useful_bytes`, and the chosen candidate's `allocated(b)` to
`*out_allocated_bytes`. Break ties between equally-good candidates by
picking the **smaller** block size.

## Example

Two tokens, sizes `{80, 96}`, `table_overhead_per_block = 16`:

- `b=64`: blocks = `ceil(80/64) + ceil(96/64)` = `2 + 2` = `4`; allocated
  = `(128 + 128) + 4*16` = `256 + 64` = `320`.
- `b=128`: blocks = `1 + 1` = `2`; allocated = `(128 + 128) + 2*16` =
  `256 + 32` = `288` — fewer wasted table entries here outweigh the
  identical padding, so `b=128` wins for this workload.

## What the gate checks

`exact_match` on the printed `(block_size, useful_bytes, allocated_bytes)`
triple for a fixed 30-token workload. Picking a candidate that isn't the
true minimizer, breaking a tie toward the larger block size instead of
the smaller one, or getting `total_blocks` wrong (e.g. counting
`floor(s/b)` instead of `ceil(s/b)`, which undercounts blocks for any
token that isn't an exact multiple of `b`) all change at least one of the
three printed numbers.
