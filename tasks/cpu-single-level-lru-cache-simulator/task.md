## Context

A **fully-associative** cache has a single set: any of its `capacity`
resident lines can live in any slot, so there is no index collision to
worry about, only capacity. **LRU** (least-recently-used) is the
standard replacement policy: on a miss with the cache already full, the
line that hasn't been touched for the longest time is the one evicted.

A byte address maps to a line by dividing out the line size:

$$
\mathrm{line} = \left\lfloor \frac{\mathrm{addr}}{\mathrm{line\_bytes}} \right\rfloor
$$

This is the foundational cache model this whole track builds on: every
harder cache task (set-associativity, blocking, replacement policy
comparisons) is a variation on this same hit/miss bookkeeping.

## Task

Implement:

```cpp
struct HitMiss { long hits; long misses; };
HitMiss simulate_lru(const long* addrs, int n, int capacity, int line_bytes);
```

Process `addrs[0..n)` in order. For each address, compute its line. If
that line is already resident, count a **hit** and mark it
most-recently-used. Otherwise count a **miss**: if the cache already
holds `capacity` lines, evict the least-recently-used one first, then
insert the new line as most-recently-used. Return the totals.

## Example

With `capacity = 3` and line trace `[0, 1, 2, 3, 0, 1, 4, 0, 1, 2, 3, 4]`:
the first four accesses (`0,1,2,3`) are compulsory misses, and inserting
`3` evicts `0` (the LRU at that point, since `capacity = 3` only holds
`1,2,3` afterward). The next `0` therefore misses again too. Working
through the whole trace this way gives `2` hits and `10` misses.

## What the gate checks

`main.cpp` runs three fixed traces through `simulate_lru`: pure
streaming over more distinct lines than fit (no reuse possible, every
access misses), a classic mixed-reuse trace with real evictions, and a
small working set that entirely fits the cache (everything after the
first pass hits). It prints `hits`/`misses` for each. The candidate's
full stdout is compared byte-for-byte (`exact_match = 1.0`) against the
reference's -- an unimplemented or mis-tracked LRU order changes some or
all of the six printed numbers.
