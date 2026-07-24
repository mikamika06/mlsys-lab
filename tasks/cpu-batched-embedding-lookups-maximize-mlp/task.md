## Context

**Memory-level parallelism (MLP)**: a real memory subsystem can have
several requests "in flight" at once — up to some finite number of
outstanding-request trackers (MSHRs) — as long as those requests don't
depend on each other. Independent loads issued back-to-back overlap their
latency; a chain of loads where each one's address depends on the *value*
loaded by the previous one (**pointer chasing**, e.g. walking a linked
list) cannot overlap at all — each hop must wait for the previous hop's
result.

A common performance bug is treating **everything** like a pointer chase:
issuing every memory access strictly one-at-a-time, even when most of them
are actually independent (e.g. gathering rows from an embedding table using
indices that are all already known — nothing about lookup #7 depends on
the *result* of lookup #3). The fix is to recognize which accesses are
truly independent and batch them into the same "wave" so they overlap,
reserving strict serialization only for the accesses that are genuinely
data-dependent.

Real hardware latency isn't reproducible across machines, so this task
uses a deterministic wave model instead (declared in `sol.hpp`): every
access gets a `wave_id`; accesses sharing a `wave_id` are considered
concurrent (cost one latency unit together), while different `wave_id`s
are serial.

## Task

The fixed workload has `NUM_CHASE_STEPS = 4` genuinely data-dependent
pointer-chase steps (`access_id` 0..3, each depending on the previous
step's loaded value) followed by `NUM_EMBED_LOOKUPS = 24` genuinely
independent embedding lookups (`access_id` 4..27, all addresses known
upfront). Implement

```cpp
void schedule_embedding_workload();
```

Call `schedule_access(access_id, wave_id)` exactly once for every
`access_id` in `[0, TOTAL_ACCESSES)`, such that:

1. Chase step `k+1`'s `wave_id` is strictly greater than chase step `k`'s
   (it cannot be issued before step `k`'s value is available).
2. No more than `MAX_WAVE_WIDTH = 8` accesses share a `wave_id`.
3. Embedding lookups have no ordering constraint of their own — pack them
   into whichever waves (including the chase-step waves) still have room,
   to **minimize the total number of distinct wave ids used**.

## Example

The naive (but valid) schedule gives every access its own wave:
`wave_id = access_id` for all 28 accesses → 28 distinct waves. The optimal
schedule uses exactly the 4 waves the chase forces, and stuffs up to 7
independent embedding lookups into each of those same 4 waves (`8 - 1` for
the chase step already occupying a slot) → `4 * 7 = 28` lookup slots, more
than enough for all 24 → 4 distinct waves total.

## What the gate checks

The driver calls `schedule_embedding_workload()` and prints
`modeled_cycles()` — the number of distinct waves used, or a huge sentinel
if rule 1 or 2 was violated. The grader compiles `solve.cpp` with
`clang++ -O2 -std=c++20`, runs it, and requires

$$ \mathrm{exact\_match} = 1.0 $$

against the reference, which prints `cycles=4`. A schedule that is merely
*valid* but treats every access as serial prints `cycles=28` — correct
rules, wrong answer, and it fails the gate.
