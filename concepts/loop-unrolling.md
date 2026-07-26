---
title: "What is loop unrolling?"
description: "Loop unrolling explained, with a measured control-operations-eliminated table you can reproduce in plain Python, plus five graded C++ exercises."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is loop unrolling?

Loop unrolling replaces a loop's one-element body with several copies of it,
so the counter increment, bound check, and branch execute fewer times for
the same work. On a 31-element loop, unrolling by 8 saves 42 of those
control operations — but unrolling by 16 gives 12 of them back, because the
leftover elements that don't fit a block of 16 still pay for their own
loop. The table below measures that reversal and names exactly where it
happens.

## How it works

A counted `for (i = 0; i < N; i++)` loop pays a fixed tax on every pass
that has nothing to do with the work inside it: increment `i`, compare it
against `N`, and branch back if the loop isn't done. Unrolling by a factor
`U` restructures the loop so one pass processes `U` elements — `i, i+1,
…, i+U-1` — before paying that tax once, shrinking the number of taxed
passes from `N` down toward `N / U`.
[Doing this to a simple elementwise update](../tasks/cpu-unroll-a-loop-by-factor-u-byte-exact/task.md)
changes nothing about the answer: `y[i] += a*x[i]` for each `i` depends on
nothing but that `i`, so the output is bit-for-bit identical no matter how
the compiler groups the iterations into blocks.

Two further effects usually motivate unrolling beyond the raw control-flow
saving. It gives the compiler more independent instructions to schedule
around latency, and — more concretely — it can break a *loop-carried
dependency*: a running total `s += x[i]` can't start iteration `i` until
iteration `i-1`'s add has retired, so an `N`-element reduction has a
serial critical path of length `N` regardless of idle execution ports.
[Splitting the reduction across several independent accumulators](../tasks/cpu-break-loop-carried-dependency-with-n-accumulators/task.md)
turns that one long chain into several short ones the CPU can overlap. The
catch for a *reduction* specifically: floating-point addition is not
associative, so changing which elements land in which accumulator can
change the rounding — the non-associativity that motivates
[kahan summation](kahan-summation.md) — even though the plain elementwise
case above stays exact regardless of grouping.

Unrolling has a hard edge case: `N` is not always a multiple of `U`. The
elements left over — `N mod U` of them — either run through their own
small scalar loop, which pays the same per-element tax the unrolled loop
was built to avoid, or get duplicated as guarded straight-line code.
Either way, that cost is easy to leave out of a back-of-envelope estimate,
and it is the entire subject of the table below.

Unrolling sits next to two loop-restructuring relatives measured elsewhere
on this site. [Cache blocking](cache-blocking.md) reorders the same kind
of loop nest for data reuse instead of for fewer branches, and the two can
combine rather than substitute for one another. [False sharing](false-sharing.md)
is a reminder that per-iteration cost can be as invisible in the source as
loop overhead is. Neither helps a loop whose limit is bandwidth rather
than control flow — that is [memory coalescing's](memory-coalescing.md)
territory on a GPU, where an unrolled loop whose lanes branch differently
is the scalar cousin of [warp divergence](warp-divergence.md).

## Control operations eliminated against unroll factor

The loop below is fixed at 31 elements — deliberately not a multiple of
2, 4, 8, or 16 — and unrolled by each of those factors in turn. What's
counted is every increment, compare, and branch the loop machinery
executes: once per full block of `U` elements, plus once more for each of
the `N mod U` leftover elements run through a scalar remainder loop.

| U | outer iterations (N÷U) | remainder | control ops | ops eliminated |
|---|---|---|---|---|
| 1 | 31 | 0 | 62 | 0 |
| 2 | 15 | 1 | 32 | 30 |
| 4 | 7 | 3 | 20 | **42** |
| 8 | 3 | 7 | 20 | **42** |
| 16 | 1 | 15 | 32 | 30 |

Reproduce it:

```bash
pip install mlsys-lab
python3 - <<'PY'
K = 2                                  # increment + compare-and-branch per iteration
N = 31                                 # not a multiple of 2, 4, 8, or 16

def control_ops(n, u):
    outer, rem = n // u, n % u
    return K * (outer + rem), outer, rem

baseline = K * N
for U in (1, 2, 4, 8, 16):
    ops, outer, rem = control_ops(N, U)
    saved = baseline - ops
    ideal_saved = K * (N - N // U)     # the tail-blind model: pretends N % U == 0
    print(f"U={U:>2}  outer={outer:>2}  rem={rem:>2}  "
          f"control_ops={ops:>3}  saved={saved:>3}  ideal_saved={ideal_saved:>3}")
PY
```

Read the five ops-eliminated numbers as one curve, not five independent
wins. It climbs from 0 to a peak of 42 at both U=4 and U=8 — doubling the
unroll factor from 4 to 8 buys nothing further, since the extra block only
trades a slightly smaller outer-loop count for a slightly bigger remainder
— then at U=16 it falls straight back to 30, tied with U=2. The remainder
loop is why: 15 of the 31 elements now run through their own scalar loop,
handing back most of what the larger factor appeared to save. A model
that ignores the remainder loop's own overhead — `ideal_saved` above, the
formula
[this bank's simplest unrolling task](../tasks/cpu-loop-overhead-ops-eliminated-by-unrolling/task.md)
uses when it fixes `N` as a multiple of `U` — never sees this: it keeps
climbing to 32, 48, 56, then 60, hiding the regression completely. The gap
between the two models is always exactly `2 × remainder` — 30 operations
at U=16, all of it bookkeeping the tail-blind estimate assumed away.

## Practise it

```bash
mlsys grade cpu-loop-overhead-ops-eliminated-by-unrolling
```

[That task](../tasks/cpu-loop-overhead-ops-eliminated-by-unrolling/task.md)
gates on `exact_match == 1.0` across five fixed `(N, U)` pairs —
`(1024,4)`, `(1000,3)`, `(7,7)`, `(50,1)`, `(17,5)` — scored with the
tail-blind formula from the reading above. The shipped starter hardcodes
`return 0`, which happens to be correct for the `(50,1)` no-unrolling case
(the right answer there really is 0) but wrong for the other four,
including `(7,7)` where the correct value is 12 and `(1024,4)` where it's
1536.

Then, in rough order:
[unroll an axpy loop by a given factor and check the output stays byte-identical regardless of U](../tasks/cpu-unroll-a-loop-by-factor-u-byte-exact/task.md),
[break a reduction's loop-carried dependency across N accumulators](../tasks/cpu-break-loop-carried-dependency-with-n-accumulators/task.md),
[model the rolled-vs-unrolled critical path of a reduction, plus its cache trace](../tasks/cpu-rolled-vs-unrolled-critical-path-with-accumulators/task.md),
[pick the unroll factor that minimises overhead against register-spill cost](../tasks/cpu-unroll-factor-minimizing-overhead-register-pressure/task.md),
and [choose how many independent accumulators to unroll-and-accumulate an attention dot product with](../tasks/cpu-ilp-of-an-attention-inner-loop-pick-unroll-accumulate/task.md).

## Common mistakes

- **Assuming savings grow monotonically with U.** At N=31 above, ops
  eliminated go 0 → 30 → 42 → 42 → 30 as U goes 1 → 2 → 4 → 8 → 16:
  unrolling 4x further, from 4 to 16, buys nothing, and unrolling 2x
  further, from 8 to 16, gives back 12 of the 42 operations already saved.
- **Sizing U from loop overhead alone, ignoring register pressure.** With
  a 997-element loop, 6 registers, and a 10-cycle spill cost, cost drops
  from 10,020 at U=6 to 10,000 at U=8 — but pushing to U=16 raises it back
  to 10,080, since two extra spilled accumulators are now paid on every
  one of the 63 remaining outer iterations.
  [Choosing U to minimise that trade-off](../tasks/cpu-unroll-factor-minimizing-overhead-register-pressure/task.md)
  means checking the register budget, not picking the largest power of two.
- **Assuming a reduction stays bit-exact under any grouping.**
  [Unrolling a pure elementwise update is exact regardless of U](../tasks/cpu-unroll-a-loop-by-factor-u-byte-exact/task.md),
  because each output depends on nothing else. Splitting a *reduction*
  across several accumulators is a different operation — floating-point
  addition is not associative, so a different grouping is, at best, equal
  only up to rounding, the same effect [kahan summation](kahan-summation.md)
  exists to control.
- **Treating the remainder as free.**
  [The byte-exact axpy task above](../tasks/cpu-unroll-a-loop-by-factor-u-byte-exact/task.md)
  sidesteps this by fixing `n` as an exact multiple of `U`; production
  code rarely gets that guarantee, and a scalar cleanup loop — or its
  branchy unrolled-remainder equivalent — is what pays for it. The
  15-element tail at U=16 in the table above is 15 elements' worth of
  exactly that cost.

## Where else to practise this

Honest comparison, from the [full survey of what exists](../LANDSCAPE.md):

- **[perf-ninja](https://github.com/dendibakh/perf-ninja)** — the closest
  general match in this track (false sharing, tiling, prefetching,
  vectorization), but its published lab list doesn't name unrolling
  specifically, and every lab there grades on a wall-clock speedup
  threshold rather than a counted number.
- **[Algorithms for Modern Hardware](https://en.algorithmica.org/hpc/)** —
  covers unrolling inside its ILP and vectorization chapters, with real
  compiler-output walkthroughs this bank doesn't attempt. Reading only: no
  exercise, no verdict.
- **[Agner Fog's optimization manuals](https://www.agner.org/optimize/)** —
  the reference for how a real compiler unrolls, spills registers, and
  schedules a pipeline once U crosses the hardware's true register count —
  the ground truth behind the register-pressure formula above. Pure
  reference, no exercises.
- **[Computer Enhance: Performance-Aware Programming](https://www.computerenhance.com/p/table-of-contents)** —
  covers this control-flow-overhead reasoning with weekly homework, but
  it's paywalled and self-checked, not auto-graded.
- **[CS:APP Cache Lab](https://csapp.cs.cmu.edu/3e/labs.html)** — shares
  this bank's deterministic-metric philosophy, but its one lab is entirely
  about cache blocking; no overlap on unrolling itself.

## References

1. Fog, A., *Optimizing Software in C++* — covers loop unrolling, register
   pressure, and instruction-level parallelism directly.
   https://www.agner.org/optimize/optimizing_cpp.pdf
2. Intel, *Intel 64 and IA-32 Architectures Optimization Reference
   Manual* — the vendor guidance on when unrolling helps and how far to
   push it before spills dominate.
   https://www.intel.com/content/www/us/en/content-details/671488/intel-64-and-ia-32-architectures-optimization-reference-manual.html
