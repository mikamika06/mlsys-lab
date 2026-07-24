## Context

A 2-level cache hierarchy (L1, L2) can enforce two different relationships
between what L1 holds and what L2 holds:

- **Inclusive**: every line in L1 must also be in L2 — L1's contents are
  always a subset of L2's. This makes cross-core coherence simple (a
  snoop only ever has to check L2), but it costs capacity: when L2 evicts
  a line to make room for a new one, it must also **back-invalidate** that
  line out of L1, even if L1's own LRU order had nothing to do with the
  eviction.
- **Exclusive**: a line lives in *at most one* level at a time. When L1
  evicts a line, instead of discarding it, it is **victim-filled** into
  L2. A line found in L2 is promoted into L1 and removed from L2. Total
  useful capacity across the hierarchy is close to $|L1| + |L2|$, instead
  of effectively just $|L2|$ (since inclusive's L1 contents are redundant
  with L2's).

Same trace, same $|L1|$ and $|L2|$, same line size — the eviction *policy*
alone changes how many accesses ultimately miss both levels and have to go
to memory.

## Task

`sol.hpp` pins the hierarchy shape: `LINE_BYTES = 64`, `L1_WAYS = 4`,
`L2_WAYS = 16` (both levels fully-associative LRU). Implement:

```cpp
void hierarchy_miss_counts(const long* addrs, int n, long* out2);
```

Run the SAME trace `addrs[0..n)` through the SAME hierarchy shape under
both policies and write `out2[0]` = miss count under **inclusive**,
`out2[1]` = miss count under **exclusive**. A "miss" is an access whose
line is in neither L1 nor L2 (must go to memory):

- **Inclusive**: on a true miss, insert the line into L2 (may evict L2's
  LRU line — if that evicted line is resident in L1, remove it from L1
  too), then insert into L1. A miss found only in L2 (L1 miss, L2 hit)
  pulls the line into L1 (whatever L1 evicts needs no further handling,
  since it's still resident in L2).
- **Exclusive**: on a true miss, insert the line into L1 only; whatever L1
  evicts to make room is inserted into L2 instead of discarded (evicting
  L2's own LRU line if L2 is full). A line found in L2 (L1 miss, L2 hit)
  is removed from L2 and inserted into L1, with the same victim-fill
  handling for whatever L1 evicts.

## Example

The driver (`main.cpp`, fixed) builds a 400-access trace over `NUM_LINES =
24` distinct cache lines (more than `L2_WAYS = 16`, so both levels feel
real eviction pressure), using a hand-rolled deterministic LCG — no
`rand()`, no wall-clock:

```
inclusive_misses=162
exclusive_misses=85
```

Inclusive's *useful* combined capacity is really just $|L2| = 16$ distinct
lines (L1's 4 lines are always a redundant subset of L2's), so it misses
almost twice as often as exclusive, whose combined capacity is close to
$|L1| + |L2| = 20$ distinct lines over the same 24-line working set.

## What the gate checks

The grader compiles `main.cpp` + your file with `clang++ -O2 -std=c++20`,
runs it, and requires both printed numbers to `exact_match` the same
driver linked against the reference simulation. Implementing only a single
shared 2-level cache (no back-invalidation, no victim-fill) makes both
counts identical — which is never what the real inclusive vs exclusive
trace produces, so the gate catches it immediately. The starter returns
`0, 0` for both.
