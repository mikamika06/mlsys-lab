## Context

PagedAttention (vLLM) stores a sequence's KV cache as a **block table**: a
list of physical block ids, one per fixed-size chunk of `block_size`
($B$) tokens. When two sequences share a common prefix — a fork, a
sampled-branch continuation, a retried request — the KV cache for that
shared prefix doesn't need to be computed or stored twice: both block
tables can point at the **same physical block ids** for the shared
portion, and only pay for fresh, private blocks once their token content
actually diverges.

For a shared prefix of length $P$ tokens, that portion occupies

$$
\left\lceil \frac{P}{B} \right\rceil
$$

logical block slots — note this rounds *up* even when $P$ isn't a
multiple of $B$: the last, only-partially-filled prefix block still
counts as one whole shared block (it can't be shared any further once
each sequence's own tail tokens begin, so it's never topped up with tail
data — each sequence's tail starts fresh in its own new block). Each
sequence's own tail, of length $\text{tail}_i = |s_i| - P$ tokens, then
needs $\lceil \text{tail}_i / B \rceil$ additional, sequence-private
blocks.

## Task

Implement `build_shared_prefix_block_tables(tokens_a, tokens_b, block_size)`:

```python
def build_shared_prefix_block_tables(tokens_a: list[int], tokens_b: list[int], block_size: int):
    ...
```

1. Compute $P$, the length of the longest common prefix of `tokens_a` and
   `tokens_b` (compare token by token; $P$ can be $0$, or as large as the
   shorter sequence's full length).
2. Build a block table for each sequence: `ceil(len(seq) / block_size)`
   entries, each a physical block id.
3. The first `ceil(P / block_size)` entries of **both** tables must be
   the exact same physical block ids (aliased). Every remaining entry,
   for each sequence independently, must be a freshly allocated id that
   is **globally unique** — never reused across the shared blocks or the
   other sequence's tail.
4. Assign physical block ids starting from `0`, in increasing order:
   first the shared prefix blocks, then sequence A's tail blocks, then
   sequence B's tail blocks.

Return `(block_table_a, block_table_b, num_physical_blocks)`, where
`num_physical_blocks` is the total count of distinct physical block ids
used across both tables.

## Example

```python
tokens_a = [1, 2, 3, 4, 5, 9, 9, 9]
tokens_b = [1, 2, 3, 4, 5, 7, 7]
build_shared_prefix_block_tables(tokens_a, tokens_b, block_size=4)
# P = 5 (shared prefix [1,2,3,4,5]); ceil(5/4) = 2 shared blocks -> ids [0, 1]
# tail_a has 3 tokens -> ceil(3/4) = 1 block -> id [2]
# tail_b has 2 tokens -> ceil(2/4) = 1 block -> id [3]
# -> block_table_a = [0, 1, 2]
#    block_table_b = [0, 1, 3]
#    num_physical_blocks = 4
```

## What the gate checks

The gate covers hand-built cases where $P$ is an exact multiple of
`block_size`, where it isn't (a partially-filled shared boundary block),
fully identical sequences, sequences with no shared prefix at all, one
sequence being a strict prefix of the other, empty sequences, and
`block_size = 1`, plus randomly generated `(tokens_a, tokens_b,
block_size)` triples from a seeded generator.

For every case the reference independently recomputes $P$ from the two
token arrays and builds both block tables with the exact same
id-assignment order described above. Your `(block_table_a,
block_table_b, num_physical_blocks)` is compared to it with exact
equality on every case. A solution that shares blocks based on a
**token-count** cutoff instead of a genuine element-by-element prefix
comparison (e.g. assuming the first `min(len(a), len(b))` tokens are
shared without checking they actually match) will pass whenever the two
inputs really do share their whole overlap, but silently over-share
whenever the sequences diverge before running out of one of them.
