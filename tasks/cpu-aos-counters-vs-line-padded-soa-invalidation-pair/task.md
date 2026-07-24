## Context

**False sharing**: when several threads on different cores write to
*different* variables that happen to live on the *same* cache line, the
cache-coherence protocol treats it exactly like real contention on a single
variable — every write from one core invalidates the line in every other
core's cache, even though no thread is actually touching another thread's
data.

The classic trigger is an **Array-of-Structs-style packed counter array**:
`int counters[N];` puts 16 consecutive `int` counters (64 bytes / 4 bytes
each) on a single 64-byte cache line. If 8 threads round-robin increment
`counters[thread_id]`, every single write (after the very first) evicts the
line from whichever core last wrote it — a full cross-core invalidation,
even though every thread only ever touches its own array slot.

The fix is **padding each counter out to its own cache line** — an
extreme, deliberately wasteful layout (64 bytes to hold one 4-byte int) that
guarantees no two threads' counters can ever share a line, so cross-core
invalidations drop to zero.

Real hardware invalidation counts aren't reproducible across machines, so
this task uses a deterministic cache-line-ownership model instead (declared
in `sol.hpp`, implemented in `main.cpp`): every 64-byte line has one
"owning" thread (whoever wrote it last); a write from a *different* thread
than the current owner is a real invalidation.

## Task

Implement

```cpp
long simulate_aos_invalidations(int num_threads, int num_increments);
long simulate_padded_invalidations(int num_threads, int num_increments);
```

For both: simulate `num_threads` threads round-robin incrementing their own
counter `num_increments` times each (thread 0, then thread 1, ..., then
`num_threads-1`, then back to thread 0, ...). For each simulated increment,
call `report_write(thread_id, byte_address)` with:

- **AoS**: `byte_address = thread_id * 4` (a packed `int counters[N]`).
- **padded**: `byte_address = thread_id * CACHE_LINE_BYTES` (one cache
  line per counter).

Call `reset_invalidations()` before the loop, and return
`total_invalidations()` after it.

## Example

With `num_threads = 2`: AoS byte addresses are `0, 4, 0, 4, ...` — both
fall in cache line 0, so every write after the first invalidates the line
(the owner alternates every write). Padded byte addresses are `0, 64, 0,
64, ...` — different lines, so line 0 is owned by thread 0 forever and line
1 by thread 1 forever: zero invalidations.

## What the gate checks

The driver calls both functions with 8 threads and 100 increments each and
prints both totals. The grader compiles `solve.cpp` with
`clang++ -O2 -std=c++20`, runs it, and requires

$$ \mathrm{exact\_match} = 1.0 $$

against the reference, which prints `aos=799` (every one of the 800 writes
except the very first invalidates the single shared line, since 8 counters
at 4 bytes each all fit within one 64-byte line) and `padded=0` (each
thread owns its own line for the entire run).
