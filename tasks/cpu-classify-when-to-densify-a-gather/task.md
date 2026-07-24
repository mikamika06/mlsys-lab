## Context

A **gather** reads elements at a list of indices directly from their source
array: `for i in 0..k: out[i] = src[idx[i]]`. If `idx` is scattered and
reused often, every repeated read can miss the cache again and again — the
classic *thrashing* gather.

The alternative is to **densify** first: make one pass that copies every
*distinct* element the workload will ever need into a small, contiguous
scratch buffer (in ascending index order, so that pass itself is as
cache-friendly as a linear scan can be), then satisfy every request by
reading from the compact buffer instead of the scattered original. Densify
pays an up-front compaction cost. It only wins when there's enough *reuse*
of a *poorly localized* index set to amortize that cost — a purely
sequential, once-each access pattern gets nothing from it, and a cluster
that's already cache-resident doesn't need it either.

The right call depends on both **locality** (do the distinct indices already
share cache lines?) and **reuse** (how many times, on average, is each
distinct index touched?) — neither one alone is enough to decide.

## Task

Implement `classify_gather_strategy(indices, k, elem_bytes)` in `solve.cpp`.
`main.cpp` owns a fixed deterministic 4-way set-associative LRU cache model
(64-byte lines, 8 sets, capacity 2048 bytes) behind `touch(byte_addr)` /
`cache_reset()`, declared in `sol.hpp`. Using only that model:

1. Simulate **GATHER**: call `cache_reset()`, then `touch(ORIG_BASE +
   indices[i] * elem_bytes)` for `i` in `0..k`, in order. Count the misses.
2. Simulate **DENSIFY**: call `cache_reset()` again, then:
   - touch `ORIG_BASE + d * elem_bytes` once for every *distinct* value `d`
     present in `indices`, in ascending order (the compaction pass);
   - touch `SCRATCH_BASE + rank * elem_bytes` for `i` in `0..k`, where
     `rank` is `indices[i]`'s 0-based position among the sorted distinct
     values (reading the request back out of the compact buffer).
   Count the total misses over both parts.
3. Return `GATHER` (`0`) if the gather miss count is `<=` the densify miss
   count, otherwise `DENSIFY` (`1`).

## Example

Indices `{0,128,256,384,512,640,768,896}` (8 values, each exactly
`8 * 64` bytes apart in a 4-byte-element array — every one aliases the
*same* cache set), cycled 10 times (80 total requests):

- **GATHER**: with 8 distinct tags round-robining through a 4-way set,
  every access evicts the block the next repeat will need — essentially
  every one of the 80 touches misses.
- **DENSIFY**: the compaction pass costs exactly 8 misses (each distinct
  source touched once, no repeats to thrash on), and the compact buffer
  (32 bytes) fits in a single line, so the 80 scratch reads cost about 1
  miss total.

Densify wins by a huge margin here. But touch that *same* 8-index set only
**once each** (no reuse) instead of cycling it, and gather wins narrowly —
the compaction pass alone already costs as much as just gathering did, and
there's no repeated reuse left to repay it.

## What the gate checks

`main.cpp` runs five fixed scenarios spanning the combinations of
locality (dense-and-line-local vs. scattered-across-cache-sets) and reuse
(cycled many times vs. touched once), and prints the five returned labels
on one line. It's compared, byte-for-byte, against the same driver linked
with `ref.cpp`: `exact_match`. A classifier that only looks at locality
(e.g. "scattered => densify") gets the no-reuse scattered scenario wrong;
one that only looks at reuse count gets the already-line-local
high-reuse scenario wrong. Only actually simulating both strategies
against the shared cache model gets all five right.
