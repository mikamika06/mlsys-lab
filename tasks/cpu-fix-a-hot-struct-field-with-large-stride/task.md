## Context

A "fat" struct — a small hot field surrounded by a lot of rarely-used
cold fields — makes its record size much bigger than a cache line. Each
record's hot field then lands in its own cache line, so scanning it once
already costs one miss per record; the real danger is scanning it more
than once. If the whole array does not fit in cache (the normal case for
real datasets), a second full pass gets **no benefit** from the first —
every line was evicted by everything scanned in between, so it pays the
same miss cost all over again.

## Task

Fix

```cpp
void hot_field_stats(long base, int stride, int hot_offset, int n, double* out);
```

`n` records of `stride` bytes each start at `base`; record `i`'s hot
field is `4` bytes at `base + i*stride + hot_offset`, with value
`(i * 37) % 97 - 48.0`. Write `sum`, `min`, `max` of the hot field into
`out[0..2]`. Every read of a hot field must go through `touch_byte()`
(see `sol.hpp`) exactly once per read — that is what the cache model
counts.

The shipped implementation gets `sum`/`min`/`max` right but scans the
records **three separate times** (once per reduction). Fix it to compute
all three reductions in a **single pass**, touching each record's hot
field only once.

## Example

With `n = 200` records of `256` bytes each (far bigger than the 4096-byte
cache model here), a single pass touches 200 distinct, never-reused
lines: 200 misses. Three separate passes over the same un-cacheable
working set each pay that cost independently: 600 misses — 3x more, for
identical `sum`/`min`/`max` results.

## What the gate checks

`exact_match`: the driver prints `sum`, `min`, `max`, and the miss count
from one fixed 200-record, 256-byte-stride trace through a 4096-byte
direct-mapped cache model. The three-separate-loop version computes the
same `sum`/`min`/`max` but a 3x higher miss count, so the printed line
differs and the match fails even though the "answer" looks right; the
single fused pass matches the reference exactly.
