## Context

A parallel binary-tree reduction ("fold N partial sums down to one")
doesn't have every thread writing every round the way a flat per-thread
counter array does. In round `r`, only threads whose id is a multiple of
`2^(r+1)` are still active — each combines its own partial sum with a
neighbour `2^r` slots away and writes the result back into its own slot.
Round 0 has half the threads writing; round 1 a quarter; and so on, until
one thread writes the final result. Crucially, **odd-indexed threads
never write again after round 0's neighbours read them** — they're
permanently retired from the write side.

That asymmetry means the naive fix for false sharing — give every
thread's slot its own full 64-byte line, the way you would for threads
that all keep writing forever — pads more than this workload actually
needs. Only the *ever-active* slots (the ones that survive to write in
later rounds) need to stay separated from each other; a tighter spacing
can still be perfectly safe if it happens to keep every pair of
same-round writers, across every round, in distinct lines.

## Task

`sol.hpp` declares `int slot_pad_bytes();`. The fixed driver models a
16-thread tree reduction: thread `tid`'s slot lives at byte address
`tid * stride`, `stride = 8 + slot_pad_bytes()`. Across the reduction's 4
rounds (8, then 4, then 2, then 1 concurrent writer — 15 writes total),
no two *different* threads' slots may ever land in the same 64-byte
line. Return the **smallest** padding for which that holds.

## Example

With `stride = 16` (`slot_pad_bytes() = 8`): round 0's writers are every
even `tid`. `line = floor(tid*16/64) = floor(tid/4)`, so `tid=0` and
`tid=2` both map to line `0` — a collision between two *different*
threads writing in the very first round. `stride = 16` is too tight.
With `stride = 32`: `line = floor(tid/2)`, which is already distinct for
every even `tid` (`0,2,4,...,14 -> 0,1,2,...,7`) — no collision, in any
round, at half the padding a flat "give everyone their own line"
approach (`stride = 64`) would use.

## What the gate checks

The driver runs the full 15-write, 4-round schedule through a
deterministic line-ownership model (no real threads, no timing) for
whatever `stride` your padding produces, and prints `stride` and the
total invalidation count. The grader compiles `solve.cpp` with
`clang++ -O2 -std=c++20`, runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{printed stride and invalidations both match the reference}
$$

The reference prints `stride=32 invalidations=0`. Padding to `stride=16`
still produces collisions (`invalidations=8`) because a tid-0/tid-2 pair
already shares a line in round 0; no padding at all (`stride=8`, 8 slots
packed into a single line) produces `invalidations=12`.
