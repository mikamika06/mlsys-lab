## Context

A real L1D is set-associative: `NUM_SETS = 64` independent pools, each
holding up to `WAYS = 8` lines (`LINE_BYTES = 64`), for `64 * 8 * 64 =
32768` bytes total. Sweeping an array with a fixed stride can be
*pathological* even when the array comfortably fits the cache's total
capacity — if the stride happens to be a multiple of some divisor of
`NUM_SETS * LINE_BYTES`, every touched line funnels into only a handful of
the 64 sets. That handful can hold more DISTINCT lines than its 8 ways, so
those lines evict each other every single sweep, while the other 63 sets
sit almost empty.

Stepping the set index by a constant `d` (mod `NUM_SETS`) visits exactly
`NUM_SETS / gcd(d, NUM_SETS)` distinct sets before repeating. If a sweep
touches `n` elements total, they split evenly across those active sets:
`n / (NUM_SETS / gcd(d, NUM_SETS))` distinct lines pile into the busiest
set. That count exceeding `WAYS` is what makes the pattern pathological.

## Task

`sol.hpp` pins the L1 shape (`LINE_BYTES=64, NUM_SETS=64, WAYS=8`).
Implement:

```cpp
int classify_pathological(long array_size, long stride);
```

For a sweep touching `0, stride, 2*stride, ...` up to `array_size` bytes
(`stride` is always a multiple of `LINE_BYTES`), let
`d = (stride / LINE_BYTES) % NUM_SETS`, `distinct_sets = NUM_SETS /
gcd(d, NUM_SETS)`, and `n = array_size / stride`. Return `1`
(pathological) if `n / distinct_sets > WAYS`, else `0` (benign).

The driver (`main.cpp`, fixed) calls your function on 6 fixed
`(array_size, stride)` pairs, and for each one ALSO runs an independent,
real `NUM_SETS`-way-indexed, `WAYS`-associative LRU simulation (sweeping
the same addresses through it twice and checking whether the second pass
adds any new misses) as ground truth — not derived from your prediction.
It prints your prediction, the empirical result, whether they agree, and
the total agreement count.

## Example

```
case=0 array_size=32768 stride=64 predicted=0 observed=0 agree=1
case=3 array_size=65536 stride=4096 predicted=1 observed=1 agree=1
```

`stride=64` (contiguous, `d=1`) spreads perfectly evenly across all 64
sets — every set gets exactly `512/64 = 8` lines, right at the `WAYS`
limit, so it's benign. `stride=4096` (`d = 4096/64 = 64 \equiv 0 \pmod
{64}`) collapses every touch into a SINGLE set — `16` lines fight over `8`
ways in that one set, so it's pathological, even though the array is only
double the cache's total capacity, not per-set overloaded everywhere.

All 6 fixed cases:

```
total_agree=6
```

## What the gate checks

The grader compiles `main.cpp` + your file with `clang++ -O2 -std=c++20`,
runs it, and requires every printed number — including your `predicted`
label for all 6 cases and `total_agree` — to `exact_match` the same
driver linked against the reference derivation. The starter always
predicts `0`: it happens to agree on the 3 already-benign cases but
disagrees on all 3 pathological ones, printing `total_agree=3` instead of
`6`, and the `predicted` values themselves diverge from the reference on
those same 3 cases either way.
