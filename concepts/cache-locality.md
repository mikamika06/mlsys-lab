---
title: "What is cache locality?"
description: "Cache locality explained, with a measured miss-rate-vs-row-length table you can reproduce without a GPU, plus a graded C++ exercise."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is cache locality?

Cache locality is the property of a memory access pattern that decides how much of a
fetched cache line actually gets reused before it is evicted, not merely how much data a
loop touches. Get it wrong and walking a matrix that comfortably fits in cache the "wrong"
way can jump its miss rate from 12.50% to a literal 100% — every single access a miss.
Below, that exact jump is measured row length by row length with a deterministic cache
model, no clock involved.

## Cache line

A cache does not fetch one element at a time; it fetches a whole **cache line** — 64
bytes here, eight `float64` values — so every count in this bank is a count of lines,
not bytes or elements. Both locality axes below are properties of that line: whether its
neighbours get used, and whether it survives long enough to be revisited.

## Spatial locality

**Spatial locality** is whether a line's neighbours get used once it is fetched: read
`a[0]` and the line brings `a[1]..a[7]` along for free, so a loop that visits them next
pays for one miss instead of eight.

## Temporal locality

**Temporal locality** is the separate question of whether an address gets revisited
before its line is evicted; a line that is still resident on a later pass is a hit no
matter how far away in the loop that revisit happens. The two axes are independent, and
the measurement below only makes sense once they are told apart.

## How it works

A 2-D array's physical layout decides which axis is the "free" one. NumPy's default is
C-contiguous — [row-major strides](../tasks/num-c-contiguous-strides-from-shape/task.md)
put the last axis at stride `elem_bytes` — while Fortran order flips it, so
[column-major strides](../tasks/num-f-contiguous-strides-from-shape/task.md) put the
*first* axis at stride 1 instead. Neither choice moves a single byte on its own:
[a transpose can swap which axis is contiguous as a pure metadata permutation, no
copy](../tasks/num-transpose-as-stride-permutation-no-copy/task.md), which is exactly why
the same buffer below gets walked two ways for the measurement — only the loop order
changes, never the storage.

The general rule, [derived algebraically rather than by
simulating](../tasks/cpu-closed-form-stride-s-line-count-vs-simulator/task.md), is that a
stride-`s` walk over `n` elements touches `n` distinct lines once `s` exceeds the line
size, and a contiguous run of about `n / (line_bytes / elem_bytes)` lines when it doesn't.
But distinct-lines-touched is only half the story, because a line touched twice is one
miss if it survived between visits and two if it didn't — which is a capacity question,
answered by [the fully-associative LRU model this whole track builds
on](../tasks/cpu-single-level-lru-cache-simulator/task.md).

Cache locality is the single-array, single-core case of two relatives covered elsewhere on
this site. [Cache blocking](cache-blocking.md) is the deliberate fix once locality is lost
— reordering a loop nest so a *tile* gets reused instead of letting a whole array cool
between visits. [Memory coalescing](memory-coalescing.md) is the same contiguous-beats-
strided idea one level down, inside a single GPU warp's 128-byte transaction. [False
sharing](false-sharing.md) is close to the opposite failure at the same line granularity —
a line shared by coincidence between cores, punished by coherence traffic rather than by
misses.

## Miss rate measured against row length

The trace walks a square `N`×`N` array of `float64`, stored row-major, once in row-major
order (`i` outer, `j` inner) and once in column-major order (`j` outer, `i` inner), and
`cachesim.simulate` counts hits and misses under the same 64-byte-line, 64-set, 8-way LRU
model used throughout this bank. Only `N` changes; it sets both the column stride in
bytes and the array's total footprint, so one sweep crosses the line-size boundary and the
cache-capacity boundary in the same run.

| N | column stride | row-major miss_rate | column-major miss_rate | column misses |
|---|---|---|---|---|
| 4 | 32 B | 0.1250 | 0.1250 | 2 |
| 8 | 64 B | 0.1250 | 0.1250 | 8 |
| 16 | 128 B | 0.1250 | 0.1250 | 32 |
| 32 | 256 B | 0.1250 | 0.1250 | 128 |
| 64 | 512 B | 0.1250 | 0.1250 | 512 |
| 127 | 1,016 B | 0.1251 | 0.1319 | 2,128 |
| 128 | 1,024 B | 0.1250 | **1.0000** | **16,384** |

Reproduce it:

```bash
pip install mlsys-lab
python3 - <<'PY'
from mlsys import cachesim

ELEM = 8

def row_major_trace(N):
    return [(i * N + j) * ELEM for i in range(N) for j in range(N)]

def col_major_trace(N):
    return [(i * N + j) * ELEM for j in range(N) for i in range(N)]

for N in (4, 8, 16, 32, 64, 127, 128):
    r = cachesim.simulate(row_major_trace(N))
    c = cachesim.simulate(col_major_trace(N))
    stride = N * ELEM
    print(f"N={N:>4}  stride={stride:>5}B  row_rate={r['miss_rate']:.4f}  "
          f"col_rate={c['miss_rate']:.4f}  col_misses={c['misses']:>6}")
PY
```

Read it in two halves. Up to `N=64` the row-major and column-major rates are identical —
the whole array fits inside the cache's 4,096-element capacity, so even though every
column step already lands in a fresh line relative to the one before it (the line-size
boundary was crossed back at `N=8`), that line was already pulled in on an earlier pass
and is still resident: losing spatial locality cost nothing as long as temporal locality
held. At `N=127` the array first exceeds that capacity and the column-major rate visibly
moves, to 0.1319 against row-major's 0.1251 — but the honest surprise is that it does not
keep degrading from there. One row length later, at `N=128`, it does not creep upward; it
jumps straight to 1.0000, 16,384 misses out of 16,384 accesses, one miss on every single
access. `128` is a power of two that lines every column step up with the cache's 64 sets,
so each line is evicted before the next pass ever returns to it — the cliff is a set-
collision, not a gradual capacity squeeze.

## Practise it

```bash
mlsys grade cpu-row-gather-vs-column-gather-embedding-traffic
```

[That task](../tasks/cpu-row-gather-vs-column-gather-embedding-traffic/task.md) gates real
C++ on `exact_match` for a fixed 5-index gather out of a 1000×32 embedding table:
`row_major=10, column_major=128` — more than 12x the memory traffic for retrieving the
identical five vectors, purely from which axis is contiguous. The shipped starter's
function body is empty, so it leaves the sentinel `out = {-1, -1}` untouched and fails
before the comparison even begins.

In roughly increasing difficulty:
[count the distinct cache lines a raw address trace touches](../tasks/cpu-distinct-cache-lines-touched/task.md),
[get the stride-1 axis into the inner loop for an arbitrary strided view](../tasks/cpu-stride-arithmetic-address-line-count/task.md),
[derive the exact stride that forces every access into one set](../tasks/cpu-derive-collision-stride-verify-vs-simulator/task.md),
and [classify a real latency-plateau curve by cache level](../tasks/cpu-classify-latency-plateaus-to-cache-levels/task.md).

## Common mistakes

- **Judging locality by "does it touch every element", not "does the line survive between
  touches".** `N=64` and `N=128` both touch every element of the array exactly once; their
  miss rates are 0.1250 and 1.0000. Coverage is not the metric — reuse is.
- **Assuming "it fits in cache" is a smooth safety margin.** `N=127` is already over
  capacity and only costs 0.1319, barely above the compulsory rate; `N=128` is one row
  length larger and costs 1.0000. The failure is a cliff at a specific alignment, not a
  slope you can eyeball from "how much over budget" the array is.
- **Confusing this with cache blocking.** [Cache blocking](cache-blocking.md) restructures
  a loop nest across *several* arrays so a shared tile gets reused; the measurement here is
  one array, one loop, two orders — the failure mode is simpler and the fix is just picking
  the right axis for the inner loop, not tiling.
- **Trusting a profile taken near, but not at, the pathological size.** Because the cliff
  at `N=128` is a set-associativity collision and not a capacity slope, a benchmark run at
  `N=100` or `N=127` reports a healthy 0.13-ish miss rate that says nothing about what
  happens at the power-of-two size one step away.

## Where else to practise this

Honest comparison, from the [full survey of what exists](../LANDSCAPE.md):

- **[Gallery of Processor Cache Effects](http://igoro.com/archive/gallery-of-processor-cache-effects/)**
  — Igor Ostrovsky's stride-vs-line-size example is the direct ancestor of the table above,
  with runnable C# rather than a reproducible number; read it for the intuition, this page
  for the exact count.
- **[perf-ninja](https://github.com/dendibakh/perf-ninja)** — real C++ labs on the same
  memory-bound topics (prefetching, alignment, tiling), graded by a wall-clock speedup
  threshold on CI hardware rather than a deterministic miss count, so the verdict here is
  the one that doesn't depend on the machine.
- **[CS:APP Cache Lab](https://csapp.cs.cmu.edu/3e/labs.html)** — the closest grading
  philosophy to this page, exact simulated miss counts against a Valgrind-traced reference,
  but one lab covering blocking and transposition, not row-vs-column sweeps specifically.
- **[Algorithms for Modern Hardware](https://en.algorithmica.org/hpc/)** — Sergey Slotin's
  free book covers cache lines, associativity and locality in more mathematical depth than
  this page attempts. No exercises, no grading — read it after the table above, not
  instead.

## References

1. Drepper, U., *What Every Programmer Should Know About Memory*, 2007 — the cache-line,
   associativity and DRAM-burst mechanics behind every number above.
   https://people.freebsd.org/~lstewart/articles/cpumemory.pdf
2. Ostrovsky, I., *Gallery of Processor Cache Effects*, 2010 — the original worked
   stride-vs-line-size and capacity-cliff demonstrations this page's table follows.
   http://igoro.com/archive/gallery-of-processor-cache-effects/
3. Fog, A., *Optimization Manuals* — microarchitecture and cache/line-size specifics for
   real x86 hardware, for readers who want the numbers this page's model approximates.
   https://www.agner.org/optimize/
