## Context

Every core has its own private cache, and the memory system keeps them
**coherent**: if core A holds a cached copy of a line and core B writes
to that same line, A's copy has to be invalidated before B's write can
be considered done — otherwise A would keep reading stale data forever.
That invalidation is a real message sent over the coherence bus/mesh,
and it costs real cycles, independent of whether A and B were ever
touching the same *variable*.

**False sharing** is what happens when two threads write to two
*different* variables that just happen to land in the same 64-byte
cache line. There's no logical data race — each thread only ever
touches its own variable — but from the coherence protocol's point of
view, every write still invalidates the *whole line*, including the
other thread's copy of its own data. Two threads incrementing separate,
unrelated counters can end up bouncing a single cache line back and
forth as if they were fighting over one variable, purely because of
memory layout.

## Task

Implement, declared in `sol.hpp`:

```cpp
long count_invalidations(const WriteEvent* trace, int n);
```

`WriteEvent` is `{int core; long addr;}` — a write by `core` to byte
address `addr`. Model a simple write-invalidate coherence protocol over
64-byte lines: maintain, per line (`addr / 64`), the set of cores
currently holding a valid cached copy (every line starts with an empty
set). For each event, in trace order:

1. Every *other* core currently in that line's owner set has its copy
   invalidated — add 1 to the running total for each one.
2. The line's owner set becomes just `{core}` (this write's core is now
   the sole holder).

Return the total invalidation count after processing all `n` events.

## Example

Two writes to the same line: core 0 writes first (owner set was empty,
0 invalidations, set becomes `{0}`), then core 1 writes (core 0 is in
the set and isn't core 1, 1 invalidation, set becomes `{1}`). A third
write by core 0 again invalidates core 1's copy — another 1. Three
writes, two invalidations, purely from alternating cores on one line.

## What the gate checks

The driver builds two 400-event traces of the exact same logical
work — 4 cores, each incrementing its own counter 100 times,
round-robin order `0,1,2,3,0,1,2,3,...` — under two different memory
layouts: **unpadded** (all 4 counters packed into 16 bytes, so every
core's writes land in the *same* 64-byte line) and **padded** (each
counter placed 64 bytes apart, one per line). It prints both
invalidation counts. The grader compiles `solve.cpp` with
`clang++ -O2 -std=c++20`, runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{both printed counts match the reference}
$$

On this fixture, unpadded measures **399** invalidations — every write
but the very first one finds a different core still holding the shared
line — while padded measures **0**: same 400 writes, same 4 cores, same
round-robin order, zero cross-core interference once each counter has
its own line.
