---
title: "What is kahan summation?"
description: "Kahan summation explained, with a measured absolute-error table against an exact math.fsum reference you can reproduce, plus a graded exercise."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is kahan summation?

Kahan summation is an algorithm for adding a sequence of floating-point numbers that
recovers most of the precision plain left-to-right addition throws away, by carrying a
second float that tracks exactly how much of each step's rounding got lost. Left
uncorrected, that loss grows with the sequence length: summing `1.0` with a million tiny
residues drifts to an absolute error of `1.000e-10` against an exact reference, while the
compensated version lands on exactly `0.0`. Below, that gap is measured across six sequence
lengths, alongside pairwise summation, the other common fix for the same failure.

## How it works

The problem is rounding, not overflow. When floating-point hardware computes `s + x`, it
forms the exact mathematical sum and then rounds it to the nearest representable value.
Once `s` is large enough that its representable values are spaced further apart than `x`,
that rounding can throw `x` away entirely — the addition runs, the result equals `s`, and
nothing records that anything happened. Summing left to right, this repeats every step once
the running total dominates the increments, and the discarded bits are gone for good.

Kahan's fix is to not let them go. Alongside the running sum `s` it keeps a second float
`c`, the compensation. At each step it computes `y = x - c` — correcting the next input for
what was lost last time — then `t = s + y`, the actual rounded addition, then recovers what
just got dropped as `c = (t - s) - y`, a floating-point identity that is itself exact
whenever `|s| >= |y|`. That condition is the catch: it is why the
[Neumaier variant](../tasks/num-kahan-vs-neumaier-when-running-sum-addend/task.md) checks
which operand is larger before computing the correction, and why
[classic Kahan without that check](../tasks/alg-kahan-neumaier-compensated-summation/task.md)
can silently discard a running sum's own bits the moment a single term outweighs it.

This is a per-element, order-dependent computation — step `k`'s correction depends on step
`k-1`'s rounded result, so there is no vectorizing it away without changing what it
computes. That is why the exercise here
[insists on a real Python loop](../tasks/num-prove-kahan-sum-is-a-real-per-element-loop/task.md)
rather than accepting `np.sum` in float64 cast back down. The sequential dependency is the
same shape of constraint as [false sharing](false-sharing.md): both are cases where the
fast, obviously-parallel rewrite is not equivalent to the slow, correct original, and the
difference only shows up once you measure rather than read the source. It is a distant
cousin of [memory coalescing](memory-coalescing.md) too — both correct for a hardware
default (rounding, DRAM bursts) that punishes one specific shape — though coalescing costs
bandwidth and Kahan costs precision, so a slow kernel and a wrong sum are different
diagnoses.

Pairwise (tree) summation attacks the same symptom differently: instead of compensating for
rounding after the fact, it shortens the chain of additions any one rounding error has to
survive, from O(N) sequential steps to about O(log N) by summing in a balanced binary tree.
It vectorizes trivially and needs no extra state, but it only slows the error's growth — it
does not, like Kahan, drive it to zero.

## Naive vs pairwise vs Kahan, measured against an exact reference

The sequence is `1.0` followed by N copies of `1e-16` — each addition on its own is far too
small to move a running total near `1.0`, so naive summation drops it, one bit at a time, N
times over. The table is the absolute error against `math.fsum`, Python's exact reference
sum, for three float64 accumulators.

| N (copies of 1e-16 after 1.0) | naive abs error | pairwise abs error | kahan abs error |
|---|---|---|---|
| 10 | 1.110e-15 | 4.441e-16 | **0.000e+00** |
| 100 | 9.992e-15 | 4.441e-16 | **0.000e+00** |
| 1,000 | 9.992e-14 | 4.441e-16 | **0.000e+00** |
| 10,000 | 1.000e-12 | 4.441e-16 | **0.000e+00** |
| 100,000 | 1.000e-11 | 6.661e-16 | **0.000e+00** |
| 1,000,000 | **1.000e-10** | 4.441e-16 | **0.000e+00** |

Reproduce it:

```bash
pip install mlsys-lab
python3 - <<'PY'
import math
from mlsys.scorers import max_abs_err

def naive_sum(vals):
    s = 0.0
    for x in vals:
        s = s + x
    return s

def pairwise_sum(vals):
    n = len(vals)
    if n <= 8:
        s = 0.0
        for x in vals:
            s = s + x
        return s
    mid = n // 2
    return pairwise_sum(vals[:mid]) + pairwise_sum(vals[mid:])

def kahan_sum(vals):
    s = 0.0
    c = 0.0
    for x in vals:
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t
    return s

for N in (10, 100, 1_000, 10_000, 100_000, 1_000_000):
    vals = [1.0] + [1e-16] * N
    exact = math.fsum(vals)
    print(N,
          max_abs_err(naive_sum(vals), exact),
          max_abs_err(pairwise_sum(vals), exact),
          max_abs_err(kahan_sum(vals), exact))
PY
```

Naive error grows in lockstep with N — each 10x increase in length costs almost exactly one
more decimal digit of accuracy, from `1.1e-15` at N=10 to `1.0e-10` at N=1,000,000, because
each of those N roundings is independent and additive. Pairwise summation breaks that
scaling: shortening the chain to O(log N) pins its error to about one ULP of the answer
(`~4e-16`) regardless of N — a thousand-fold win over naive at the largest size tested, but
not zero. Kahan reaches exactly `0.0` at every length, because in this sequence the running
sum's magnitude is always larger than the increment being added, precisely the regime the
compensation identity is exact in. **That clause is also where it breaks**: reorder the
values so a term larger than the running sum arrives mid-sequence — `[1.0, 1e16, 1.0, 1.0,
-1e16, 1.0, 1.0, 1.0]`, true sum `6.0` — and classic Kahan returns `5.0`, a whole unit lost,
because `(t - s) - y` only recovers rounding when `s` is the dominant operand.

## Practise it

```bash
mlsys grade num-kahan-summation-beats-naive-fp32
```

[That task](../tasks/num-kahan-summation-beats-naive-fp32/task.md) gates two things at once
on a fixed, shuffled 3,002-element float32 array: `improvement_ratio >= 100` — your
compensated sum's error against a float64 reference must beat plain float32 summation by at
least 100x — and `loop_events >= 3000`, a line-tracer confirming the array was walked
element by element. The obvious failure mode is what the second gate exists to catch: cast
the array to float64 and call `np.sum`, and `improvement_ratio` passes trivially while
`loop_events` stays near zero, because the exercise is about recovering precision *inside*
float32, not escaping it.

In increasing variety:
[pairwise tree summation](../tasks/num-pairwise-tree-summation/task.md) (`rel_err <= 1e-9`
against `np.sum`),
[Kahan in real C++](../tasks/cpu-kahan-pairwise-summation-accuracy/task.md) (compiled by
`clang++`, gated on `max_abs_err <= 1e-3` against a fixture where naive drifts by almost
1,000 from the true total),
[fitting the log-log error slope](../tasks/num-error-growth-naive-vs-kahan-vs-pairwise/task.md)
of all three methods directly,
[the same recurrence as a single-thread CUDA kernel](../tasks/gpu-sequential-vs-pairwise-vs-kahan-summation-error/task.md),
and [Kahan over fp16 inputs](../tasks/llm-kahan-compensated-summation-vs-naive-fp16/task.md),
where the storage format itself is losing the bits.

## Common mistakes

- **Using classic Kahan when a term can outgrow the running sum.** The worked example above
  returns `5.0` against a true sum of `6.0` — a full unit lost, 16.7% relative error — the
  moment `1e16` arrives while `s` is still `1.0`. Neumaier's magnitude check fixes this, at
  the same O(1) memory cost.
- **Trusting pairwise summation where a gate demands an exact match.** The table above shows
  pairwise holding at `~4e-16`, never `0.0`, at every N tested — real progress, but not
  compensation. A bit-for-bit check like the C++ task's still fails against it, where
  Kahan's `0.000e+00` passes.
- **Vectorizing the recurrence away.** `c` at step `k` depends on `t` at step `k-1`, so a
  `np.cumsum`-shaped rewrite computes a different, wrong quantity, not a faster version of
  the same one. The harness for
  [proving it's a real loop](../tasks/num-prove-kahan-sum-is-a-real-per-element-loop/task.md)
  traces Python line events and requires `line_count >= 1000`, which no vectorized shortcut
  can produce.
- **Expecting compensation to recover precision the storage format never had.** Rerun the
  experiment above with every value cast to float32, and naive, pairwise, and Kahan all
  report the identical `1.000e-10` error at N=1,000,000 — measured, not assumed. `1e-16`
  relative to `1.0` sits about nine orders of magnitude below float32's `~1.19e-07` ULP, so
  there is nothing left in that budget to compensate for;
  [Kahan over low-precision inputs](../tasks/sys-compensated-kahan-summation-in-low-precision/task.md)
  is built around exactly this boundary.

## Where else to practise this

Honest comparison, from the [full survey of what exists](../LANDSCAPE.md):

- **[Tensor Puzzles](https://github.com/srush/Tensor-Puzzles)** — 21 auto-checked NumPy
  puzzles including a `sum` reimplementation, but purely as a broadcasting exercise; it does
  not touch floating-point precision, so it will not flag a wrong sum for the reasons this
  page is about.
- **[100 NumPy exercises](https://github.com/rougier/numpy-100)** — the canonical
  self-checked list, organized around "how do I do X" rather than "why did the answer round
  this way"; loosely overlapping, and ungraded.
- **[Float Exposed](https://float.exposed/)** — the best hands-on tool for *why* rounding
  happens: type a decimal, watch its exact bit pattern and the ULP gap to its neighbors. No
  exercises, but it makes this page's mechanism concrete rather than algebraic.
- **The survey's own verdict is blunt here**: across everything catalogued for numerics, "no
  exercise set anywhere checks a learner's Kahan/compensated-summation implementation
  against a tolerance" — this bank's Kahan-adjacent tasks across Python, C++, CUDA, and fp16
  fill a gap nothing else in the survey filled.

## References

1. Kahan, W., *Pracniques: Further Remarks on Reducing Truncation Errors*, Communications of
   the ACM, 8(1), 1965. https://doi.org/10.1145/363707.363723
2. Higham, N. J., *The Accuracy of Floating Point Summation*, SIAM Journal on Scientific
   Computing, 14(4), 1993. https://doi.org/10.1137/0914050
3. Goldberg, D., *What Every Computer Scientist Should Know About Floating-Point Arithmetic*,
   ACM Computing Surveys, 23(1), 1991. https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
