## Context

Each core's cache holds data at 64-byte-line granularity — the coherence
protocol has no idea two 8-byte counters packed into the same line
belong to *different* threads. If thread 0's counter sits at byte 0 and
thread 1's sits at byte 8, they share a line, and every time either
thread writes its own counter, the coherence protocol invalidates the
other core's copy of that whole line — even though the two threads never
touch each other's data. This is **false sharing**: contention that has
nothing to do with the algorithm, purely a consequence of memory layout.

The fix doesn't require synchronization or redesigning the algorithm —
just spacing each thread's counter out to its own cache line so the
coherence protocol never has a reason to invalidate across threads.

## Task

`sol.hpp` declares `int counter_pad_bytes();`. The fixed driver models 8
threads doing 5 rounds of round-robin counter writes (thread 0, then 1,
..., then 7, repeated 5 times — 40 writes total), where thread `tid`'s
counter lives at byte address `tid * stride` and
`stride = 8 + counter_pad_bytes()`. Implement `counter_pad_bytes()` to
return enough padding that `stride` becomes a multiple of `64` — so no
two threads' counters ever fall in the same 64-byte line, for any number
of threads or rounds.

## Example

With `counter_pad_bytes() = 0`, `stride = 8`: 8 threads' counters sit at
bytes `0, 8, 16, ..., 56` — all inside the single line `[0, 64)`. Every
write after the very first flips that line's owner to a different
thread, so 39 of the 40 writes in the driver's schedule are
invalidations. With `counter_pad_bytes() = 56`, `stride = 64`: the 8
counters sit at `0, 64, 128, ..., 448` — 8 different lines, never
shared, so no write is ever an invalidation, no matter how many rounds
run.

## What the gate checks

The driver computes `stride`, runs the 8-thread/5-round schedule through
a deterministic line-ownership model (no real threads, no timing), and
prints `stride` and the total invalidation count. The grader compiles
`solve.cpp` with `clang++ -O2 -std=c++20`, runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{printed stride and invalidations both match the reference}
$$

The reference prints `stride=64 invalidations=0`. Returning `0` padding
leaves `stride=8`, and the repeated round-robin schedule racks up
`invalidations=39` — the gate catches it immediately, and the effect
gets worse (not better) the more rounds a real workload would run.
