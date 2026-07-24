## Context

A normal ("temporal") store to memory that misses the cache does a
**write-allocate**: the CPU first fetches the old line into cache (a
*read-for-ownership*), then overwrites part of it, on the assumption
you'll touch that line again soon. **Non-temporal (NT) stores** skip
all of that — they write straight to memory and never allocate a cache
line at all.

That trade cuts both ways:

- If you're about to **re-read** what you just wrote, temporal stores
  are a win: the data is already resident, so the reread is free. An NT
  store throws it away, so the reread has to fetch it from memory all
  over again.
- If you're **not** going to reuse it soon (a one-shot fill, a copy
  destination, a render target you're about to hand off) — or if the
  write is simply too big to fit in cache alongside data that's still
  useful — a temporal store buys you nothing but still pays full price:
  it evicts whatever else was cached to make room for lines you'll
  never read back. An NT store leaves that other data alone.

Which effect wins depends on both how big the write is relative to the
cache **and** whether the written data gets reread — there's no single
rule that ignores one of those two facts.

## Task

Implement, declared in `sol.hpp`:

```cpp
bool nt_stores_help(long working_set_bytes, bool reused_soon);
```

`sol.hpp` also declares (already defined in `main.cpp`, yours to call,
not to implement) a fixed experiment:

```cpp
long run_workload(long working_set_bytes, bool use_nt, bool reused_soon);
```

which — against a shared deterministic 8192-byte cache — warms a small
2048-byte "hot" region `H`, writes `working_set_bytes` of a separate
buffer using temporal stores (`use_nt=false`) or non-temporal stores
(`use_nt=true`), then re-reads `H` (counting any misses caused by the
buffer write evicting it) and, if `reused_soon` is true, also re-reads
the buffer itself. It returns the total miss count from those re-reads
— the real, measured cost of that store strategy for that scenario.

`nt_stores_help` must call `run_workload` twice with the *same*
`working_set_bytes` and `reused_soon` — once with `use_nt=false`, once
with `use_nt=true` — and return `true` iff the non-temporal run's cost
is strictly lower.

## Example

For `working_set_bytes=2048, reused_soon=true`: the buffer is small
enough to sit in cache alongside `H` without evicting it. Temporal
storing costs `0` (both `H` and the buffer stay resident, everything
rereads as a hit); non-temporal storing costs `32` (the buffer was
never cached, so its reread misses once per line). Temporal wins here —
`nt_stores_help` returns `false`.

For `working_set_bytes=32768, reused_soon=true`: the buffer is 4x the
cache's capacity. Temporal storing costs `544` (it evicts `H`, and
keeps re-evicting its own earlier lines as it overflows the cache,
leaving most of itself uncached by the time the write finishes);
non-temporal storing costs `512` (no eviction of `H` — its cost is
purely the compulsory refetch of the buffer on reread). Non-temporal
wins — `nt_stores_help` returns `true`.

## What the gate checks

The driver runs 10 scenarios — 5 buffer sizes (`2048, 4096, 8192,
16384, 32768` bytes, spanning below, at, and above the 8192-byte cache)
crossed with `reused_soon in {false, true}` — and prints `nt_helps` for
each. The grader compiles `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{every printed verdict matches the reference build}
$$

On this fixture the reference measures `nt_helps=0` for the two
smallest sizes (`2048`, `4096` — always, regardless of `reused_soon`)
and `nt_helps=1` for `8192` and above — even when `reused_soon=true`,
because once the buffer alone matches or exceeds the cache's capacity,
temporal storing can no longer keep it resident anyway, so paying for
write-allocation buys nothing.
