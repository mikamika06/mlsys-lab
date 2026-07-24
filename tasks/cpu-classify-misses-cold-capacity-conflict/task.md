## Context

Not all cache misses are equal, and the classic "3C" model gives each one a
cause:

- **Cold** (compulsory): the line has never been touched before. No caching
  policy in the world avoids this one — it is the price of the first read.
- **Capacity**: the working set genuinely does not fit in the cache. Even a
  cache with unlimited associativity (any line can go in any slot) would
  have evicted it by now.
- **Conflict**: the working set *would* fit if placement were unrestricted,
  but this cache maps addresses into a fixed number of sets, and too many
  live addresses landed in the same set and evicted each other.

The standard way to tell capacity and conflict apart is to run the trace
through a second, hypothetical **fully-associative** cache with the same
total number of lines (one set, that many ways) alongside the real one. If
an address is not cold and the real cache misses:

- the fully-associative cache misses too $\Rightarrow$ **capacity** (nothing
  could have saved this line — the working set is just too big),
- the fully-associative cache hits $\Rightarrow$ **conflict** (unlimited
  associativity would have kept it cached — the real cache's fixed
  set-mapping is the only reason it's gone).

## Task

Implement, in `solve.cpp`:

```cpp
struct MissCounts { int cold; int capacity; int conflict; };
MissCounts classify_misses(const uint64_t* addrs, int n,
                            int line_bytes, int sets, int ways);
```

Given a trace of `n` byte addresses and cache geometry (`line_bytes`, `sets`,
`ways`), build two fresh LRU caches from scratch (no state may persist
across calls to `classify_misses`):

1. the real cache: `sets` sets of `ways` ways, line index
   `addrs[i] / line_bytes`, set index `line % sets`;
2. a fully-associative cache with the same total capacity: `sets * ways`
   ways in a single set.

Replay the trace through both, in lockstep, one access at a time. For every
access that misses in the real cache, classify it:

- **cold** if this line's address has never appeared earlier in the trace
  (check this independently of either cache's state — an address can be
  "cold" and still evict something on its way in);
- otherwise **capacity** if the fully-associative cache also misses;
- otherwise **conflict** (the fully-associative cache hit).

Real-cache hits are not counted in any of the three buckets. Both caches use
LRU eviction.

## Example

The driver (`main.cpp`, fixed) pins the geometry at 64-byte lines, 4 sets, 2
ways (8 lines / 512 bytes total) and runs 4 traces:

- **A** — 20 distinct lines, single pass, never repeated: every access is a
  first touch, so `cold=20 capacity=0 conflict=0`.
- **B** — 3 addresses that all hash to the *same* set, accessed round-robin
  4 times (12 accesses total). All 3 fit easily in an 8-line
  fully-associative cache, but only 2 of them fit in their shared 2-way set
  at once: `cold=3 capacity=0 conflict=9`.
- **C** — 16 distinct lines spread evenly across all 4 sets, swept twice (32
  accesses). 16 lines do not fit in an 8-line cache under *any* placement
  policy: `cold=16 capacity=16 conflict=0`.
- **D** — the three shapes above concatenated (disjoint address ranges) into
  one trace, exercising all three categories together.

The starter always returns `{0, 0, 0}`, which is wrong on every trace except
a trivially empty one.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires the printed `cold=/capacity=/conflict=` line for every
trace to match the reference (`main.cpp` + `ref.cpp`) byte-for-byte
(`exact_match == 1.0`). Skipping the fully-associative side-simulation (e.g.
labeling every non-cold miss as "capacity") gets trace A right but produces
the wrong split on B, C, and D; forgetting to track "ever seen" independent
of cache residency miscounts cold on every trace with repeats.
