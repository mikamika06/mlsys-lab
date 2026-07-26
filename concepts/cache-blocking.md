---
title: "What is cache blocking?"
description: "Cache blocking explained, with a measured misses-vs-tile-size table you can reproduce without a GPU, plus a graded C++ exercise."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is cache blocking?

Cache blocking is the technique of reordering a loop nest so that it finishes
all the work on one small tile of the data — small enough to fit in cache —
before moving to the next tile, instead of sweeping the whole array once per
outer iteration. Get the tile size right and a matrix multiply's cache misses
can drop more than 55-fold; get it one step too large and the "blocked"
kernel can end up missing *more* than the unblocked one. Below is that whole
curve, measured tile by tile by a deterministic cache model.

## How it works

A cache holds a small window of memory near the processor; a loop that reads
or writes a whole matrix touches far more addresses than that window holds,
so by the time the loop comes back to reuse a value, its line is gone and has
to be fetched again. The naive triple-nested `ijk` matmul is the textbook
case: for every row `i`, the inner `j, k` loops sweep `B[k][j]` with a
column-major stride across a matrix that may be many times larger than the
cache, and that sweep restarts from cold for every one of the `N` values of
`i`. [Ranking `ijk` against `ikj` and `jki` by miss count](../tasks/cpu-rank-ijk-ikj-jki-matmul-by-misses/task.md)
shows that reordering those same three loops, with no blocking at all, can
change the miss count by more than 3x on its own.

Blocking fixes this without changing what gets computed — only the order
addresses are visited in. Split each of the three index ranges into chunks
of size `T` and finish every `(i, j, k)` triple inside one `T × T × T` cube
before advancing to the next cube. While one cube is live, the three
`T × T` tiles of `A`, `B`, and `C` it touches are small enough to stay
resident, so every element loaded gets reused `T` times instead of once.
[Blocking a transpose](../tasks/cpu-blocked-cache-aware-transpose-misses/task.md)
is the same idea applied where the stride can't be removed, only confined to
a tile.

The catch is that "small enough" is a hard capacity condition, not a
preference. Three live tiles have to fit together —
`3 · T² · elem_bytes ≤ cache_bytes` — and the moment `T` crosses that line
the tiles start evicting each other mid-cube, so the benefit disappears all
at once rather than degrading gently.
[Deriving the largest `T` that still fits](../tasks/cpu-pick-tile-b-for-capacity-fits-flag/task.md)
and [fixing a formula that forgets two of the three tiles](../tasks/cpu-fix-a-too-large-tile-max-b-for-l2/task.md)
both gate on getting that arithmetic right — the common bug solves for one
tile's capacity alone and returns a `T` about `√3` times too large.

Cache blocking sits next to two relatives covered elsewhere on this site.
[Memory coalescing](memory-coalescing.md) is the same "contiguous beats
strided" idea one level down, inside a single GPU warp's 128-byte
transaction rather than a whole loop nest. [False sharing](false-sharing.md)
is close to the opposite failure at the same cache-line granularity — a line
shared by coincidence and punished for it, rather than deliberately reused.
And once a tile size is chosen,
[verifying the tiled and naive matmuls agree](../tasks/cpu-verify-tiled-matmul-equals-naive/task.md)
is the check that blocking only changed the address order, never the
arithmetic result.

## Misses measured against tile size

The table tiles a 128×128 double-precision matmul (`3 · 128³ = 6,291,456`
array touches in total) at six tile sizes and runs every touch through the
same 64-byte-line, 64-set, 8-way LRU model `cachesim` uses everywhere else in
this bank. Only `T` changes; the visit order inside a cube is always
`i → j → k`, touching `A`, `B`, then `C` once each.

| tile T | misses | miss_rate |
|---|---|---|
| 1 | 2,118,688 | 0.3368 |
| 8 | 40,432 | 0.0064 |
| 16 | **37,672** | **0.0060** |
| 32 | 609,600 | 0.0969 |
| 64 | 2,121,792 | 0.3372 |
| 128 | 2,118,688 | 0.3368 |

Reproduce it:

```bash
pip install mlsys-lab
python3 - <<'PY'
from mlsys import cachesim

N, ELEM = 128, 8                      # 128x128 float64 matrices
BASE_A, BASE_B, BASE_C = 0, N * N * ELEM, 2 * N * N * ELEM

def tiled_trace(T):
    addrs = []
    for ii in range(0, N, T):
        for jj in range(0, N, T):
            for kk in range(0, N, T):
                for i in range(ii, ii + T):
                    for j in range(jj, jj + T):
                        for k in range(kk, kk + T):
                            addrs.append(BASE_A + (i * N + k) * ELEM)
                            addrs.append(BASE_B + (k * N + j) * ELEM)
                            addrs.append(BASE_C + (i * N + j) * ELEM)
    return addrs

for T in (1, 8, 16, 32, 64, 128):
    r = cachesim.simulate(tiled_trace(T))
    print(f"T={T:>3}  misses={r['misses']:>8}  miss_rate={r['miss_rate']:.4f}")
PY
```

Read the table as one curve, not six independent numbers. `T=1` is
degenerate blocking — a one-element tile gives no reuse — so it reproduces
the naive sweep's 2,118,688 misses exactly, and `T=128` (one tile covering
the whole matrix) reproduces the identical number for the identical reason:
both extremes visit every address in the same order. Misses fall by two
orders of magnitude by `T=16`, the best point measured here, then turn
around hard: by `T=32` they are back above 600,000, and by `T=64` —
2,121,792 — the "blocked" kernel is measurably *worse* than not blocking at
all. That reversal arrives earlier than a naive capacity check predicts:
three 32×32 tiles of 8-byte doubles are 24,576 bytes, comfortably under this
cache's 32,768-byte capacity, but 8-way associativity means `A`, `B`, and
`C`'s power-of-two-strided addresses land in the same sets and evict each
other before the capacity line is reached — the same conflict-miss effect
the original blocking literature (reference 1) measured on real hardware
three decades ago.

## Practise it

```bash
mlsys grade cpu-tile-ijk-to-hit-an-l2-miss-gate
```

[That task](../tasks/cpu-tile-ijk-to-hit-an-l2-miss-gate/task.md) gates real
C++ on `exact_match` over a pair of miss counts, fixed at a 48×48 matmul:
`naive_misses=4735`, `tiled_misses=1147`. The shipped starter's function
body is empty — it touches nothing, so it fails before a miss count is even
compared. The more interesting failure comes after that: tiling only one or
two of the three loops, or touching an `(i, j, k)` triple more than once
inside a cube, changes the printed count with no compiler error and no wrong
matmul result, because the gate compares memory-access counts, not the
answer. The counter is [`src/mlsys/sim/cache.py`](../src/mlsys/sim/cache.py),
the same deterministic model `cachesim.simulate` calls above.

In roughly increasing difficulty:
[derive the resident tile size for a given L2 budget](../tasks/cpu-pick-tile-keeping-working-set-in-l2/task.md),
[compute the optimal tile size under a traffic model](../tasks/num-optimal-tile-size-under-cache-constraint/task.md),
[compare naive-vs-tiled reuse distances against L2](../tasks/cpu-naive-vs-tiled-matmul-reuse-distances-fit-l2/task.md),
and [a cache-oblivious recursive matmul that needs no `T` at all](../tasks/cpu-cache-oblivious-recursive-matmul-miss-count/task.md).

## Common mistakes

- **Sizing the tile from one array's capacity, not three.** Solving
  `T² · elem_bytes ≤ cache_bytes` instead of `3 · T² · elem_bytes ≤
  cache_bytes` gives a `T` about `√3` too large — for a 32,768-byte L2 that's
  90 instead of 52, and three 90×90 tiles are 97,200 bytes, nearly three
  times over budget.
- **Tiling only some of the loops.** Blocking `i` and `j` but leaving `k`
  as a full sweep still walks the entire uncached dimension every time;
  the code looks tiled and the miss count says it isn't.
- **Trusting the capacity formula alone.** `T=32` above satisfies
  `3 · T² · elem_bytes ≤ cache_bytes` with room to spare and still costs
  609,600 misses — sixteen times the `T=16` optimum — because associativity
  conflicts bite before capacity does. The formula is necessary, not
  sufficient; a simulator or a profiler is what confirms the tile actually
  stays resident.
- **Assuming any blocking beats no blocking.** `T=64` here misses more than
  `T=1`. Outside the window where the fitting math holds, tiling adds
  bookkeeping for a benefit that has already turned negative.

## Where else to practise this

Honest comparison, from the [full survey of what exists](../LANDSCAPE.md):

- **[CS:APP Cache Lab (CMU 15-213)](https://csapp.cs.cmu.edu/3e/labs.html)** —
  the closest topic match found anywhere: its second half is literally
  optimizing a matrix-transpose kernel, graded on exact Valgrind-traced miss
  counts against a fixed reference, the same deterministic-count philosophy
  used above. One lab out of eleven, and the handout is unchanged since
  roughly 2015.
- **[perf-ninja](https://github.com/dendibakh/perf-ninja)** — has a real
  loop-tiling/interchange lab in the same real-C++-on-real-hardware style as
  this bank's tasks, but its pass/fail is a wall-clock speedup threshold on
  CI hardware, so the verdict depends on the machine — exactly what the
  miss-count gate here avoids.
- **[Algorithms for Modern Hardware — Matrix Multiplication](https://en.algorithmica.org/hpc/algorithms/matmul/)** —
  a genuinely good worked case study of blocking a real GEMM down through
  L1/L2/L3 with numbers from real hardware. Nothing to submit, no grading;
  read it once the exercises here pass and you want the "why" behind
  register blocking, not just tile sizing.
- **[Gallery of Processor Cache Effects](http://igoro.com/archive/gallery-of-processor-cache-effects/)** —
  Ostrovsky's stride-vs-line-size example is the single-array cousin of the
  three-array problem measured above. Still a blog post to read, not an
  exercise with a verdict.

## References

1. Lam, M. S., Rothberg, E. E., Wolf, M. E., *The Cache Performance and
   Optimizations of Blocked Algorithms*, ASPLOS 1991.
   https://doi.org/10.1145/106972.106981
2. Slotin, S., *Algorithms for Modern Hardware* — "Matrix Multiplication",
   the blocking/tiling case study this page's tile-size arithmetic follows.
   https://en.algorithmica.org/hpc/algorithms/matmul/
3. Drepper, U., *What Every Programmer Should Know About Memory*, 2007 —
   the cache-blocking case study on the CPU side.
   https://people.freebsd.org/~lstewart/articles/cpumemory.pdf
