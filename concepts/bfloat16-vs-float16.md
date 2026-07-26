---
title: "What is bfloat16 vs float16?"
description: "bfloat16 vs float16 explained, with a measured table of exponent/mantissa bits and the exact overflow boundary for each 16-bit format you can reproduce, plus a graded exercise."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is bfloat16 vs float16?

Bfloat16 vs float16 is a choice of how to spend the same 16-bit budget: bf16 gives 8 bits to
the exponent and 7 to the mantissa, fp16 gives 5 to the exponent and 10 to the mantissa. That
split moves fp16's largest finite value down to 65,504 while bf16's stays at 3.39×10³⁸, the
same ceiling as full float32. Below is exactly where each format's numbers run out, measured
by casting real values down and watching for infinity.

## How it works

An IEEE binary float is three fields: one sign bit, an exponent field that sets the magnitude,
and a mantissa field that sets the precision within that magnitude. Both fp16 and bf16 spend
16 bits total, so every bit given to one field is a bit taken from the other — there is no way
to have both fp16's precision and bf16's range in 16 bits, only a choice of which one you need
more. This is a fixed-budget trade-off in the same shape as
[trading DRAM traffic for shared-memory capacity in memory coalescing](memory-coalescing.md) or
[trading cache capacity for coherence traffic when padding against false sharing](false-sharing.md):
spend the budget on one property, and you pay for it in the other, and the machine will not
lend you bits it does not have.

fp16 is the older IEEE-754 half-precision format: 10 mantissa bits give it roughly three
decimal digits of precision, but its 5-bit exponent (bias 15) tops out at 65,504 before the
next step is infinity. bf16 was defined later, specifically for machine learning training,
by keeping float32's exact exponent field and bit width and simply dropping float32's bottom
16 mantissa bits. That is why a fp32→bf16 conversion is a bit-shift — the task
[FP32 to bfloat16 round-to-nearest-even encoding](../tasks/num-fp32-bf16-truncate-rne-round-trip/task.md)
is exactly that operation — while a fp32→fp16 conversion has to re-derive a completely
different, narrower exponent range and can genuinely overflow values that fp32 represented
fine.

That overflow risk is the practical reason bf16 dominates training today: activations,
attention logits, and gradient norms in a deep network routinely swing across many orders of
magnitude, and any one of them landing above 65,520 silently turns an fp16 tensor's value
into `inf` — which is why fp16 mixed-precision training needs loss scaling as a workaround and
bf16 training does not. The two formats do cost the same in memory: both are 2 bytes per
element, a fact the task
[bf16 Memory Footprint Ratio with Bounded Error](../tasks/num-bf16-memory-footprint-ratio-with-bounded-error/task.md)
gates on directly, so the choice between them is invisible to any byte-counting argument and
entirely about which failure mode — silent overflow, or coarser rounding — a given tensor can
tolerate. fp32 sits underneath both as the format they are each truncating or re-deriving from,
which is why it belongs in the same table rather than a separate page.

## Exponent, mantissa, and the overflow boundary, measured

The only thing varied below is the dtype. For each one this counts the exponent and mantissa
bit widths, reads off the largest finite value and the smallest normal value, and finds —
by casting a `float64` value down and checking for `inf` — the exact value at which each
format stops being finite.

| format | exponent bits | mantissa bits | max finite | smallest normal | eps | first value that overflows |
|---|---|---|---|---|---|---|
| float16 | 5 | 10 | 6.550400e+04 | 6.103516e-05 | 9.765625e-04 | 6.552000e+04 |
| bfloat16 | 8 | 7 | 3.389531e+38 | 1.175494e-38 | 7.812500e-03 | 3.396177e+38 |
| float32 | 8 | 23 | 3.402823e+38 | 1.175494e-38 | 1.192093e-07 | 3.402824e+38 |

Reproduce it — `float16` and `float32` come from NumPy's own dtypes; `bfloat16` is not a
NumPy dtype, so this uses `ml_dtypes.bfloat16`, the real extension type with an 8-bit
exponent and 7-bit mantissa, rather than a manual float32-truncation stand-in:

```bash
pip install mlsys-lab ml_dtypes
python3 - <<'PY'
import numpy as np
import ml_dtypes

def overflow_boundary(cast, max_finite):
    """Smallest float64 x such that casting x down to `cast` yields inf."""
    lo, hi = np.float64(max_finite), np.float64(max_finite) * 1.0000001
    with np.errstate(over="ignore"):
        while not np.isinf(float(cast(hi))):
            hi *= 1.0000001
        for _ in range(100):
            mid = (lo + hi) / 2
            if np.isinf(float(cast(mid))):
                hi = mid
            else:
                lo = mid
    return float(hi)

specs = [
    ("float16",  np.float16,         np.finfo(np.float16)),
    ("bfloat16", ml_dtypes.bfloat16, ml_dtypes.finfo(ml_dtypes.bfloat16)),
    ("float32",  np.float32,         np.finfo(np.float32)),
]
for name, dt, fi in specs:
    fmax, ftiny, feps = float(fi.max), float(fi.tiny), float(fi.eps)
    boundary = overflow_boundary(dt, fmax)
    print(f"{name:9s} exp={fi.iexp} mant={fi.nmant:2d} "
          f"max={fmax:.6e} tiny={ftiny:.6e} eps={feps:.6e} "
          f"overflow_at={boundary:.6e}")
PY
```

Read the `eps` column as the precision story: fp16's epsilon is roughly 8× smaller than
bf16's, so for anything inside its narrow range fp16 is the more faithful representation —
exactly why it stays attractive for inference. Read `max finite` and `first value that
overflows` as the range story: bf16 and float32 differ from each other by less than one part
in 1,700, because bf16 is float32's own exponent field with nothing removed from it, while
fp16's ceiling sits 47 orders of magnitude lower. The place this breaks a real model is the
gap between `max finite` and the overflow boundary — 16 for fp16, tiny in relative terms — a
gap so narrow that a value only slightly past the format's own advertised maximum silently
becomes non-finite rather than saturating or raising.

## Practise it

```bash
mlsys grade num-find-fp16-overflow-bf16-survives-boundary
```

[That task](../tasks/num-find-fp16-overflow-bf16-survives-boundary/task.md) gates on
`exact_match == 1.0` against a reference boundary found by the same binary-search idea used
above. The shipped starter raises `NotImplementedError`, but its own docstring names the trap
it is built to catch: returning `65504.0` — fp16's own maximum finite value — looks like a
correct answer and is not one, because 65504 does not overflow to `inf`; it is the largest
value that doesn't. The boundary the gate wants is strictly above it.

In increasing scope:
[encode fp32 into fp16 and bf16 bit patterns](../tasks/sys-ieee-fp16-bf16-encode-decode/task.md)
(gates `byte_exact_fraction` at `1.0` on both),
[round-trip reconstruction error across fp16, bf16, and fp8](../tasks/num-per-format-reconstruction-error-fp16-bf16-fp8/task.md),
[classify which of a batch of values overflow fp16 while surviving in bf16](../tasks/llm-which-values-overflow-fp16-but-not-bf16/task.md),
[do the same classification from inside a CUDA kernel](../tasks/gpu-bf16-vs-fp16-dynamic-range-classification/task.md),
and, once the format facts are solid,
[measure how fp16 accumulation error grows relative to fp32](../tasks/cpu-fp16-vs-fp32-accumulation-error-growth/task.md)
(`rel_err <= 1e-06`), which is where the mantissa difference actually costs you something at
runtime instead of just at the boundary.

## Common mistakes

- **Assuming `np.bfloat16` exists.** Plain NumPy has no bfloat16 dtype — `np.float16` and
  `np.float32` are built in, bf16 is not — so code that writes `np.bfloat16(x)` will raise
  `AttributeError` unless `ml_dtypes` (or a framework that registers the dtype, like
  TensorFlow or JAX) is imported first.
- **Truncating fp32 to fp16 the way you truncate it to bf16.** bf16 conversion is a bit-shift
  because the exponent field is unchanged; fp16 conversion has to renormalize into a 5-bit
  exponent, and values outside that range don't truncate, they overflow.
- **Treating "16-bit float" as one format.** The two share a bit width and nothing else in
  practice: fp16 buys precision at a range of 65,504, bf16 buys 38 orders of magnitude more
  range at 8× coarser steps. Neither number describes "half precision" in general.
- **Reading `eps` as the error on every value.** Machine epsilon is the spacing at `1.0`;
  the true step size scales with the exponent, so the absolute rounding error on a value near
  bf16's maximum is roughly 2¹²⁰ times larger than the error near 1.0, not the same `eps`.

## Where else to practise this

From the [full survey of what exists](../LANDSCAPE.md) for this area — floating-point
bit-level exploration is unusually well served by two independent tools, but neither one
grades you:

- **[Float Exposed](https://float.exposed/)** — flip individual bits of a half, bfloat16,
  float, or double and see the exact decimal value and the delta to the next representable
  number. The best hands-on tool for building intuition about the table above; it has no
  exercises to check yourself against.
- **[fp-conv](https://sw23.github.io/fp-conv/)** — the same click-a-bit interaction extended
  to the full modern ML dtype zoo: fp8 e4m3/e5m2, fp6, fp4, tf32, and custom bit layouts.
  Covers formats this page doesn't touch at all; also purely a visualizer.
- **[100 NumPy exercises](https://github.com/rougier/numpy-100)** — touches dtype and casting
  incidentally as general array practice, with no automated grader and no focus on
  floating-point range or rounding specifically.
- The landscape survey's own verdict for this area: nobody grades a Kahan-summation
  implementation, a stable-softmax overflow case, or a dtype-promotion prediction against real
  NumPy rules — that sharper half of numerics has no graded competitor anywhere, which is
  what the tasks linked above are for.

## References

1. IEEE Standard for Floating-Point Arithmetic (IEEE 754-2019).
   https://ieeexplore.ieee.org/document/8766229
2. Google Cloud, *BFloat16: The secret to high performance on Cloud TPUs*.
   https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus
3. `ml_dtypes` — the extension NumPy dtypes for bfloat16 and the ML fp8 formats.
   https://github.com/jax-ml/ml_dtypes
