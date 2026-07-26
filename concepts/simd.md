---
title: "What is SIMD?"
description: "SIMD explained, with a measured vector-ops-vs-tail-ops table you can reproduce without a GPU or an intrinsics header, plus a graded C++ exercise."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is SIMD?

SIMD is a CPU execution mode in which one instruction operates on several
data elements packed into a single register, instead of one element per
instruction. An 8-wide add does the work of eight scalar adds in one
issue, but only for the elements that fill the register — a 37-element
loop vectorized at width 16 still needs 5 separate scalar instructions for
what does not fit. Below, exact instruction counts across five widths and
two loop lengths show where that arithmetic stops paying off.

## What is SIMD in computer architecture?

SIMD is one of four classes in Flynn's taxonomy, which classifies
computer architectures by instruction and data streams: SISD is one
instruction on one data element, an ordinary scalar core; SIMD is one
instruction on many data elements, this page; MIMD is many independent
instructions each on their own data, what separate CPU cores give you. A
modern chip stacks both — MIMD across cores, SIMD inside each core's own
registers — answering different parallelism questions.

## How it works

SIMD (Single Instruction, Multiple Data) packs `W` values into one
register and applies one instruction to all of them at once. A scalar
loop `for i: c[i] = a[i] + b[i]` becomes a
loop that loads `W` elements of `a` and `b` into a register each, adds the
two registers once, and stores `W` results back — one instruction where
there used to be `W`.

That win depends on the loop's length dividing evenly by `W`, and real
arrays rarely cooperate. Every vectorized loop therefore compiles to two
pieces: a *main loop* running full-width vector instructions over
`(n / W) * W` elements, and a *tail*, scalar or masked, over the
`n % W` elements left over. Getting the tail wrong is not a slowdown but
a correctness bug — the leftover elements are never written, exactly the
failure
[the tail-masking task](../tasks/cpu-tail-masking-for-n-not-multiple-of-width/task.md)
gates on.

Whether a loop vectorizes at all is a compiler decision, not a language
feature: the compiler has to prove the iterations are independent, and
pointer aliasing, a data-dependent trip count, or a loop-carried
dependency can each block it. Once inside the vector unit, a data-dependent branch
in the loop body does not disappear, it becomes a *mask*: every lane
executes both sides and a per-lane select keeps the right result, so a
rarely-taken branch is paid for on every lane, every iteration — the same
lane-level cost as [warp divergence](warp-divergence.md) on a GPU, billed
as masked work instead of serialized re-execution.

The width itself is a property of the datatype, not a fixed constant:
moving data from float32 to [bfloat16 or float16](bfloat16-vs-float16.md)
or to an [8-bit quantized integer](integer-quantization-ranges.md) doubles
or quadruples how many elements fit in the same register, which is why
quantized inference kernels lean on SIMD heavily. Getting operands into
those registers efficiently is [cache blocking](cache-blocking.md) and
[memory coalescing](memory-coalescing.md)'s job, since a vector load still
costs a whole cache line and a strided gather defeats whatever width was
gained by widening. Collapsing a vector register's lanes back into one
scalar reintroduces the reordering questions [Kahan
summation](kahan-summation.md) answers, since a horizontal reduction sums
`W` partial results in a different order than the scalar loop did.

## SSE, AVX2, AVX-512, NEON: which SIMD instruction set gives which width

The `W` column above isn't free-floating: on real x86 CPUs it's fixed by
which SIMD instruction set the compiler targets. SSE is 128-bit, four
32-bit floats per register — the `W=4` row. AVX2 is 256-bit, eight
floats — `W=8`. AVX-512, where implemented, is 512-bit, sixteen floats —
the table's widest row, `W=16`. On Arm, NEON is 128-bit, the same width and
lane count as SSE, with no 256- or 512-bit tier in mainstream use. `W=2`
matches no 32-bit instruction set on either architecture; it's in the
table to show the trend, not because a 64-bit vector unit ships in
hardware.

## SIMD vectorization: vector ops, tail ops and total ops against width and N

The table below varies the vector width `W` over two fixed loop lengths —
`N=37`, chosen so it divides evenly by none of 2, 4, 8 or 16, and `N=5`,
chosen to be shorter than the two largest widths — and counts, per width,
how many full-width vector instructions run, how many scalar tail
instructions run, and the total. `reduction` is `N / total_ops`: the
factor by which vectorizing actually cut the instruction count, against
the `W`-fold cut it promises.

| N | width W | vector ops | tail ops | total ops | reduction |
|---|---|---|---|---|---|
| 37 | 1 | 37 | 0 | 37 | 1.00x |
| 37 | 2 | 18 | 1 | 19 | 1.95x |
| 37 | 4 | 9 | 1 | 10 | 3.70x |
| 37 | 8 | 4 | 5 | 9 | 4.11x |
| 37 | 16 | 2 | 5 | **7** | 5.29x |
| 5 | 1 | 5 | 0 | 5 | 1.00x |
| 5 | 2 | 2 | 1 | 3 | 1.67x |
| 5 | 4 | 1 | 1 | 2 | 2.50x |
| 5 | 8 | **0** | 5 | 5 | 1.00x |
| 5 | 16 | **0** | 5 | 5 | 1.00x |

Reproduce it:

```bash
python3 - <<'PY'
WIDTHS = (1, 2, 4, 8, 16)

def op_counts(n, w):
    vector_ops = n // w
    tail_ops = n % w
    total_ops = vector_ops + tail_ops
    return vector_ops, tail_ops, total_ops

for n in (37, 5):
    for w in WIDTHS:
        v, t, tot = op_counts(n, w)
        speedup = n / tot
        print(f"n={n:>3} width={w:>2} vector_ops={v:>2} tail_ops={t:>2} "
              f"total_ops={tot:>2} speedup={speedup:.2f}x")
PY
```

Read the `N=37` rows as the honest version of the promise: widening from
8 to 16 only moves the reduction from 4.11x to 5.29x, nowhere near
doubling, because the tail is fixed at 5 scalar instructions regardless of
`W` while only the vector portion shrinks. The `N=5` rows show the sharper
failure — the break-even point for a width `W` is exactly `N = W`: below
it, `vector_ops` is 0 no matter how wide the register is, so the
"vectorized" path issues precisely the same instruction count as the
`W=1` scalar loop. At `N=5`, that has already happened for both `W=8` and
`W=16` — their rows are bit-for-bit identical to the scalar row. **Any
loop shorter than the register width cannot pay for vectorizing it**,
because not one full-width instruction ever executes.

## Practise it

```bash
mlsys grade cpu-tail-masking-for-n-not-multiple-of-width
```

[That task](../tasks/cpu-tail-masking-for-n-not-multiple-of-width/task.md)
gates real C++ (`clang++ -O2 -std=c++20`) on `max_abs_err <= 1e-06`
against a reference `vec_add`. The shipped starter runs the `WIDTH=4`
main loop correctly but never adds the tail loop counted above — for
`n=22` (`22 % 4 = 2`), the last two output elements keep a
`-999.0` sentinel instead of their real values, an error of roughly 1029
against the reference — enough to fail the gate before vector correctness
is even in question.

In roughly increasing difficulty:
[predict which of 5 loops auto-vectorize](../tasks/cpu-predict-which-of-5-loops-auto-vectorize/task.md) (no code),
[elements per register for fp32, fp16 and int8](../tasks/cpu-elements-per-register-for-fp32-fp16-int8/task.md) (no code),
[lane utilization of a masked op](../tasks/cpu-simd-lane-utilization-for-a-masked-op/task.md),
[the modeled op-count speedup of a vectorized dot product](../tasks/cpu-modeled-op-count-speedup-of-vectorized-dot/task.md) —
the same idea as the table above, applied to a reduction — and
[a real Arm NEON intrinsic kernel](../tasks/cpu-neon-intrinsic-elementwise-kernel/task.md),
graded on byte-exact output and a cache-miss budget together.

## SIMD optimization: common mistakes

- **Expecting the reduction to scale with the width.** Doubling `W` from
  8 to 16 at `N=37` only moves the reduction from 4.11x to 5.29x, not to
  8.22x — the 5-element tail costs 5 instructions regardless of `W`, and
  that fixed cost, not `N`, sets the floor.
- **Dropping the tail loop entirely.** It looks like a performance
  shortcut and is a correctness bug, as [the tail-masking
  task](../tasks/cpu-tail-masking-for-n-not-multiple-of-width/task.md)'s
  starter shows above.
- **Vectorizing a loop shorter than the register width.** At `N=5`,
  widths 8 and 16 both give `total_ops=5`, same as never vectorizing —
  `vector_ops` is 0 whenever `N<W`.
- **Assuming the compiler vectorized because the code "looks"
  vectorizable.** Auto-vectorization needs proof of no aliasing and no
  loop-carried dependency; overlapping reads and writes, or an
  unreassociated accumulator, compile to scalar code.

## Where else to practise this

Honest comparison, from the [full survey of what exists](../LANDSCAPE.md):

- **[perf-ninja](https://github.com/dendibakh/perf-ninja)** — a real
  vectorization lab among its 20+, on real hardware like this bank's
  tasks, but graded by a wall-clock threshold on CI hardware
  (Alderlake/Zen3/M1) — the verdict depends on the machine, which the
  instruction-count gates here avoid.
- **[Computer Enhance: Performance-Aware Programming](https://www.computerenhance.com/p/table-of-contents)**
  — Casey Muratori's paid course has a unit on SSE intrinsics with
  homework, self-checked against community solutions on GitHub; no
  autograder.
- **[Algorithms for Modern Hardware](https://en.algorithmica.org/hpc/)** —
  Sergey Slotin's free book has a dense SIMD-intrinsics chapter with real
  case studies, but no exercises; read it after these tasks, for
  hand-tuned intrinsics instead of counted instructions.
- **[Agner Fog's optimization manuals](https://www.agner.org/optimize/)**
  — not a competing exercise resource, but the reference this page's model
  doesn't replace: exact per-instruction latency and throughput on real
  microarchitectures, which an op count never predicts.

## References

1. Intel, *Intrinsics Guide* — the reference for every x86 SIMD
   instruction, indexed by width and datatype.
   https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
2. Fog, A., *Optimizing software in C++* — covers when a compiler will and
   will not auto-vectorize a loop, and why.
   https://www.agner.org/optimize/optimizing_cpp.pdf
3. Arm, *Neon Intrinsics Reference*, for the equivalent instruction set on
   Arm hardware. https://developer.arm.com/architectures/instruction-sets/intrinsics/
