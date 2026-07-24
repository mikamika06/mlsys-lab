## Context

For a fully-associative cache with a fixed capacity, **Belady's OPT (MIN)**
algorithm is the provably optimal replacement policy: on every miss with a
full cache, evict whichever resident line is referenced **furthest in the
future** — or never referenced again, which counts as "furthest" of all.
No online policy (LRU, FIFO, ...) can beat OPT's miss count on the same
reference string, because OPT gets to see the whole future. That's exactly
why it isn't implementable on real hardware — but it's the standard
yardstick every real policy is measured against.

## Task

Implement

```cpp
int belady_opt_misses(const int* refs, int n, int capacity);
```

Simulate a fully-associative cache of `capacity` lines over the reference
string `refs[0..n)`:

1. If `refs[i]` is already resident: a **hit**, nothing changes.
2. Otherwise: a **miss**. If there's a free slot, fill it. If the cache is
   full, evict whichever resident page's *next occurrence* (strictly after
   index `i`) is furthest away — a page with no future occurrence at all
   beats any page that does recur.

Return the total number of misses.

## Example

Capacity 2, reference string `[A, B, C, A, B]`:
- `A`: miss (empty slot) — cache `{A}`.
- `B`: miss (empty slot) — cache `{A, B}`.
- `C`: miss, cache full. `A`'s next use is index 3; `B`'s next use is
  index 4. Evict `B` (furthest next use) — cache `{A, C}`.
- `A`: hit.
- `B`: miss (not resident) — 4 misses total.

## What the gate checks

`main.cpp` builds a deterministic 50-reference string over a working set
of 6 page ids (mostly cycling, with occasional jumps) run against a
4-line cache, calls your function, and prints the miss count. The grader
compiles your `.cpp` with the real local `clang++`, runs it, and requires
the printed count to match the reference's exactly
($\mathrm{exact\_match}=1.0$). A greedy policy like "evict slot 0" or a
policy that only looks at *past* reuse (LRU) is a well-known, valid cache
strategy — but it isn't OPT, and on this reference string it produces a
different (higher) miss count.
