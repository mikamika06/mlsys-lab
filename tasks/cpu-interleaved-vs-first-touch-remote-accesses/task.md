## Context

On a NUMA (non-uniform memory access) machine, RAM is physically split
across nodes, and a CPU reading from its *own* node's memory is faster
than reading from a *remote* node's. The OS decides which node backs
each 4KB page under one of two common policies:

- **First-touch**: a page isn't backed by any node until something
  actually accesses it — at that moment, whichever thread touched it
  first "wins" the page, permanently, for that page's whole lifetime.
  This is fast when each thread mostly works on its *own* private data:
  the first (and only) toucher is always the eventual owner, so private
  data is local by construction.
- **Interleaved**: pages are assigned round-robin across nodes the
  moment memory is allocated, before anyone has touched anything —
  page $p$ always belongs to node $p \bmod \text{num\_nodes}$,
  regardless of who ends up using it.

First-touch is the default almost everywhere because most real
workloads are dominated by private, per-thread data. But it has a sharp
failure mode: if one region is genuinely **shared** and some thread
happens to sweep through it first, that thread's node quietly becomes
the home for the *entire* region — every other thread's access to it is
now remote, for the rest of the program.

## Task

Implement, declared in `sol.hpp`:

```cpp
void count_remote_accesses(const Access* trace, int n, int num_nodes,
                            long* first_touch_remote, long* interleaved_remote);
```

`Access` is `{int thread; long addr;}`. Thread `t` is pinned to node
`t % num_nodes`. For the *same* trace, compute, under **each** policy,
how many of the `n` accesses are remote (accessing thread's node !=
that access's page's home node under that policy — page number is
`addr / PAGE_BYTES`, `PAGE_BYTES = 4096`):

- **First-touch remote count**: track, per page, which node touched it
  first *in trace order* (the first access to a page is always local to
  itself, by definition — it's the one that sets the home). Every
  later access to that page is remote iff its thread's node differs
  from that recorded home.
- **Interleaved remote count**: a page's home is always
  `(addr / PAGE_BYTES) % num_nodes`, needing no history — remote iff
  the accessing thread's node differs from that.

Write the two totals into `*first_touch_remote` and
`*interleaved_remote`.

## Example

Page 5 is touched by thread 2 first, then thread 0, then thread 2
again, with `num_nodes = 4`. First-touch: thread 2's first access sets
the home to node 2 (local); thread 0's access is remote (node 0 != 2);
thread 2's second access is local again (node 2 == 2) — 1 remote out of
3. Interleaved: page 5's home is fixed at `5 % 4 = 1` regardless of
touch order — thread 2 (node 2, remote), thread 0 (node 0, remote),
thread 2 (node 2, remote) — 3 remote out of 3, even though two of the
three accesses came from the thread that "owns" the data in practice.

## What the gate checks

The driver builds a 320-access trace over 4 threads / 4 nodes: each
thread touches its own exclusive 16-page private block (4 times per
page, 256 accesses total, never touched by any other thread), plus a
separate 16-page shared block that every thread sweeps once, in thread
order 0→1→2→3 (64 accesses). It prints both counts. The grader compiles
`solve.cpp` with `clang++ -O2 -std=c++20`, runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{both printed counts match the reference}
$$

On this fixture, first-touch measures **48** remote accesses — zero
from the private blocks (each is only ever touched by its own thread),
all 48 from the shared block (thread 0's sweep claims every shared
page, so threads 1-3's sweeps, 48 accesses, are entirely remote).
Interleaved measures **240** — the private blocks alone contribute 192
remote accesses, because assigning pages round-robin has no way to know
those pages are exclusively used by one thread. Same trace, same node
count, a 5x difference from nothing but which policy decided page
placement.
