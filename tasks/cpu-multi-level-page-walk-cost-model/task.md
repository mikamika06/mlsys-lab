## Context

Translating a virtual address that isn't in the TLB means walking a
4-level page table (PML4 → PDPT → PD → PT on x86-64) — up to four
sequential memory reads just to find the fifth, real one. Hardware
softens this with a **page-walk cache (PWC)**: each level keeps its own
small cache of recently-resolved table entries, so a walk that reuses a
table another nearby translation already touched skips straight to a
cheap cache hit at that level instead of paying a full memory round
trip.

The cost of one walk is the sum of four independent per-level
hit-or-miss decisions plus the final data access — and because each
level's PWC has finite capacity, whether a given lookup hits depends on
what that *specific level's* cache has resident, which itself depends on
the whole sequence of addresses walked before it. Level 0 (PML4) usually
stays resident for huge stretches of addresses; level 3 (PT) churns
constantly.

## Task

Implement

```cpp
long page_walk_cycles(const int* keys, int num_addrs, const int* cap,
                       long hit_cycles, long miss_cycles, long data_cycles);
```

Maintain 4 independent, fully-associative LRU caches, one per page-table
level (`0..3`), with capacities `cap[0..3]`. Process `num_addrs`
addresses in order; address `j`'s per-level table keys are
`keys[j*4 + i]` for `i` in `0..3`. For each address, walk levels `0` to
`3`: at level `i`, if `keys[j*4+i]` is already resident in level `i`'s
cache, add `hit_cycles` and mark it most-recently-used; otherwise add
`miss_cycles`, insert it as most-recently-used (evicting that level's
least-recently-used key first if it's already at `cap[i]`). After the 4
levels, add `data_cycles` for the final access. Return the grand total
over every address.

## Example

Two addresses with keys `(0,0,0,0)` then `(0,0,0,1)`, `cap = {1,1,1,1}`,
`hit_cycles=4`, `miss_cycles=120`, `data_cycles=4`: address 1 is a cold
walk — 4 misses (`480`) + data (`4`) = `484`. Address 2 shares levels
0–2's keys with address 1 (still resident, all hits: `12`), but level 3's
key changed (`0` → `1`, a miss: `120`) + data (`4`) = `136`. Total:
`484 + 136 = 620`.

## What the gate checks

The driver walks a fixed, structured sequence of 120 addresses (a nested
`2 x 3 x 4 x 5` sweep over the four levels' keys — level 0 changes every
60 addresses, level 3 every address) through deliberately tight
capacities `cap = {1, 2, 2, 3}` (each smaller than that level's distinct
key count, so real evictions happen, not just cold misses), with
`hit_cycles=4`, `miss_cycles=120`, `data_cycles=4`, and prints the total.
The grader compiles `solve.cpp` with `clang++ -O2 -std=c++20`, runs it,
and requires

$$
\mathrm{exact\_match} = 1 \iff \text{printed total\_cycles matches the reference}
$$

The reference computes `total_cycles=20032`. A stub that returns `0`
fails immediately; one that sums the 4 levels' costs but forgets to
model each level's LRU eviction correctly (e.g. always treating a key as
a miss, or never evicting) produces a different total and fails too.
