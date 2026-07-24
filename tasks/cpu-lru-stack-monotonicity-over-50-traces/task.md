## Context

LRU has a property no other common replacement policy guarantees: the
**stack (inclusion) property**. Run the same access trace against two
fully-associative LRU caches of size `k` and `k+1`. At every point in the
trace, the set of lines resident in the size-`k` cache is a *subset* of
the lines resident in the size-`k+1` cache — a bigger cache never forgets
anything a smaller one under LRU would remember. One direct consequence:
**miss count is a non-increasing function of cache size**. Growing the
cache can never make LRU miss *more* — cache-size-vs-miss-rate curves for
LRU are always flat-or-falling, never bumpy. (Some replacement policies,
famously FIFO, don't have this property at all — that's Bélády's
anomaly — which is exactly why the property is worth checking rather
than assuming.)

## Task

Implement

```cpp
long lru_miss_count(const int* ids, int n, int capacity);
```

which simulates a fully-associative LRU cache holding at most `capacity`
distinct line ids, processing the trace `ids[0..n)` (ids may repeat — a
repeat means re-accessing an already-resident line). On each access: if
the id is currently resident, it's a HIT and becomes most-recently-used;
otherwise it's a MISS, and the id is inserted as most-recently-used,
evicting the current least-recently-used resident id first if the cache
is already full. Return the total number of misses over the trace.

## Example

Trace `[1, 2, 1, 3, 1]`, `capacity = 2`: access `1` (miss, cache `{1}`),
`2` (miss, cache `{2,1}`), `1` (hit — resident — cache becomes
`{1,2}`), `3` (miss, evicts LRU `2`, cache `{3,1}`), `1` (hit). Total: 3
misses. With `capacity = 3`, nothing ever gets evicted and `3` is a miss
too but `1`'s hits stay hits — 3 misses again here, but in general a
bigger capacity's miss count can only be `<=` a smaller one's, never `>`.

## What the gate checks

The driver generates 50 fixed traces (200 accesses each, ids drawn from
a fixed-seed deterministic generator — same 50 traces every run, no
`rand()`/no clock) and, for each trace, runs your `lru_miss_count` at six
capacities `{1, 2, 4, 8, 16, 32}`. It counts how many of the 50 traces
have non-increasing miss counts across those six capacities, and it also
sums every miss count computed across all 50 traces × 6 capacities. It
prints both numbers. The grader compiles `solve.cpp` with `clang++ -O2
-std=c++20`, runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{both printed numbers match the reference}
$$

The reference reports `monotonic_count=50` (the stack property holds for
every trace, as it must for a correct LRU simulation) and
`total_miss_sum=50754`. A stub that returns `0` unconditionally
"passes" the monotonicity check by accident (`0 >= 0` trivially holds
across all six capacities, so `monotonic_count` still comes out `50`) —
but `total_miss_sum=0` doesn't match the reference's real total, so the
gate still catches it.
