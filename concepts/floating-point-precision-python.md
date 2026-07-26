---
title: "What is floating point precision python?"
description: "Floating point precision python explained, with a measured table of 0.1's exact stored fraction, the ULP gap across magnitudes, and the exact float64 value where +1 stops registering, plus a graded exercise."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is floating point precision python?

Floating point precision python is fixed at 53 significant bits per `float` value —
IEEE-754 binary64's 52 stored mantissa bits plus one implicit leading bit — the same
regardless of magnitude or how many decimal digits get printed. That fixed budget is
why `0.1 + 0.2` evaluates to an error of `5.551115123125783e-17` against `0.3`, and why past
`9007199254740992.0` a `float64` stops registering `+ 1.0` at all. Below, 0.1's exact stored
fraction, the size of that error at six different magnitudes, and that exact threshold are
all measured and reproduced below using nothing but the standard library.

## How it works

A Python `float` is a C `double`: one sign bit, an 11-bit exponent, and a 52-bit stored
mantissa, for 53 significant bits once the implicit leading `1` is counted back in. That budget
is fixed — it does not grow or shrink with the size of the number — so the *absolute* gap
between one representable value and its neighbour, the unit in the last place or ULP, scales
with magnitude while the *relative* gap stays flat at roughly `2^-52`. This is the same
trade every fixed-width format makes, just with a different split of the bits:
[bfloat16 vs float16](bfloat16-vs-float16.md) spends 16 bits on exponent-vs-mantissa instead
of 64, and the [int8 / int4 / int16 ranges](integer-quantization-ranges.md) drop the exponent
entirely for a uniform grid, trading dynamic range for a simpler, denser one.

Most decimal fractions have no exact binary64 representation for the same reason `1/3` has no
finite decimal one: the target base's prime factors don't divide the source's. `0.1` is stored
as the nearest representable binary64 value, not the mathematical `1/10` — and Python's `repr`
prints the shortest decimal string that round-trips back to that same stored bit pattern, which
is `0.1`, not the (much longer) exact value underneath it. `fractions.Fraction(0.1)` and
`decimal.Decimal(0.1)` both convert *from* the stored float, so both surface the same exact
binary value rather than fixing anything — the fix is constructing from a string or an integer
ratio instead, which is exactly what the measurement below does.

Two consequences of that fixed 53-bit budget come up constantly in numeric code. First, error
accumulates when many roundings compound in the same direction, which is the failure
[Kahan summation](kahan-summation.md) exists to correct by tracking what each addition drops.
Second, precision runs out from the other end too: subtracting two close large numbers, or
`exp()`-ing a large logit before summing, discards the low bits the same way, which is why
[log-sum-exp](log-sum-exp.md) reorganizes the arithmetic rather than computing it directly.
Neither problem is a bug in Python; both are the direct, measurable cost of spending only 53
bits on every value regardless of what it needs.

## 0.1's exact value, the ULP gap by magnitude, and the +1 threshold

Three things are varied below: the representation of `0.1` (float repr, exact fraction, exact
decimal), the magnitude at which the ULP gap is measured, and the exponent at which adding
`1.0` to a `float64` value stops changing it.

**0.1's exact stored value**

| representation | value |
|---|---|
| `repr(0.1)` | `0.1` |
| `Fraction(0.1)` | `3602879701896397 / 36028797018963968` |
| `Decimal(0.1)` (from the float) | `0.1000000000000000055511151231257827021181583404541015625` |
| `Decimal("0.1")` (from the string) | `0.1` |

**ULP gap by magnitude**

| magnitude | ULP (absolute gap) | ULP / magnitude |
|---|---|---|
| 0.1 | 1.387779e-17 | 1.387779e-16 |
| 1.0 | 2.220446e-16 | 2.220446e-16 |
| 100.0 | 1.421085e-14 | 1.421085e-16 |
| 1,000,000.0 | 1.164153e-10 | 1.164153e-16 |
| 1,000,000,000,000.0 | 1.220703e-04 | 1.220703e-16 |
| 1e+16 | 2.000000e+00 | 2.000000e-16 |

**Where `+ 1.0` stops registering, and the cost of exactness**

| quantity | value |
|---|---|
| `0.1 + 0.2 - 0.3` | `5.551115123125783e-17` |
| smallest `float64` where `x + 1.0 == x` | `9007199254740992.0` (= 2^53) |
| its predecessor, `pred + 1.0 == pred`? | `9007199254740991.0` -> `False` |
| `sys.getsizeof(0.1)` | 24 bytes |
| `sys.getsizeof(Decimal("0.1"))` | 104 bytes |
| `sys.getsizeof(Fraction(1, 10))` | 48 bytes |

Reproduce it:

```bash
pip install mlsys-lab
python3 - <<'PY'
import sys, math
from decimal import Decimal
from fractions import Fraction

diff = 0.1 + 0.2 - 0.3
print("0.1 + 0.2 - 0.3 =", diff)

f = Fraction(0.1)
print("Fraction(0.1)        =", f.numerator, "/", f.denominator)
print("Decimal(0.1)          =", Decimal(0.1))
print("Decimal('0.1')        =", Decimal("0.1"))

for m in (0.1, 1.0, 100.0, 1_000_000.0, 1_000_000_000_000.0, 1e16):
    u = math.ulp(m)
    print(f"ulp({m!r:<17}) = {u:.6e}   ulp/mag = {u / m:.6e}")

x = 2.0 ** 53
pred = math.nextafter(x, 0.0)
print("2**53          =", x, " x+1.0==x        :", x + 1.0 == x)
print("predecessor    =", pred, " pred+1.0==pred  :", pred + 1.0 == pred)

print("sizeof(float 0.1)          =", sys.getsizeof(0.1))
print("sizeof(Decimal('0.1'))     =", sys.getsizeof(Decimal("0.1")))
print("sizeof(Fraction(1, 10))    =", sys.getsizeof(Fraction(1, 10)))
PY
```

The ULP/magnitude column is the whole story in one number: it sits at `~2.2e-16` (`2^-52`,
`float64`'s machine epsilon) no matter whether the magnitude is `0.1` or `1e16`, because
doubling the exponent doubles the absolute gap between neighbours by exactly the same factor
the value itself grew by. `9007199254740992.0` (`2^53`) is the point that relative constancy
runs into: it is the first `float64` value whose own ULP reaches `2.0`, so adding `1.0` — half
a step — rounds back to where it started, and every larger finite `float64` inherits the same
fate, since the ULP only grows from there. `Decimal(0.1)` shows the fraction table's real point:
converting *from* a float exposes the exact binary value already stored, `0.1000...055511...`,
not the decimal `0.1` a reader typed — only `Decimal("0.1")` or `Fraction(1, 10)`, built from
outside the float, are actually exact, and that exactness costs 2x–4.3x the bytes of the
`float` it replaces.

## Practise it

```bash
mlsys grade num-exact-rational-value-of-each-float
```

[That task](../tasks/num-exact-rational-value-of-each-float/task.md) gates three metrics on a
fixed sweep of floats spanning zero, subnormals, and the full exponent range:
`fields_exact_fraction == 1.0` and `ratio_exact_fraction == 1.0` (your bit-field decode and
your exact rational must match a `fractions.Fraction`-based oracle on every input), plus
`max_rel_err < 1e-9` computed in exact big-integer arithmetic before any float conversion. The
constraint that catches shortcuts: it must be built from `struct.pack`/`struct.unpack` and
integer arithmetic alone — no `Fraction(x)`, no `x.as_integer_ratio()` — so passing means you
reconstructed the value the hardware computes, not called the library that already does it.

In increasing variety:
[report fp32 machine epsilon as its raw bit pattern](../tasks/num-report-machine-epsilon-for-fp32/task.md)
(`exact_match == 1.0`),
[derive epsilon for fp32, fp16, and bf16 by bit-stepping from 1.0](../tasks/num-derive-eps-for-fp32-fp16-bf16-by-bit-stepping-from-1-0/task.md)
(`rel_err <= 1e-12`),
[verify the 2^23 spacing law by counting representable float32 values in a range](../tasks/num-verify-the-2-23-spacing-law-by-counting-representables/task.md),
[compute `nextafter(x, +inf)` as a pure integer bit increment](../tasks/num-nextafter-x-inf-by-integer-bit-increment/task.md),
and [measure ULP distance between two values as an integer step count](../tasks/num-ulp-distance-as-integer-step-count/task.md)
— all `exact_match == 1.0`, all built from the same bit-pattern arithmetic the primary task
requires.

## Common mistakes

- **Comparing floats with `==` instead of a tolerance.** The measured
  `0.1 + 0.2 - 0.3 = 5.551115123125783e-17` is not a bug to patch around case by case; it is
  the fixed ULP gap at magnitude `1.0` (`2.220446e-16`) showing up exactly where the rule
  predicts. `abs(a - b) <= tol` is the fix, not chasing individual failing comparisons.
- **Assuming `Decimal(a_float)` fixes the precision.** The table's own
  `Decimal(0.1) = 0.1000000000000000055511151231257827021181583404541015625` is the same
  binary64 value `0.1` already was, just printed with more digits — the float's imprecision,
  not a cure for it. Only `Decimal("0.1")`, built from the string, or `Fraction(1, 10)`, built
  from an exact ratio, are actually exact.
- **Reading machine epsilon as the error bound at every value.** `2^-52` is the *relative*
  spacing at every magnitude, but the *absolute* one measured above ranges from `1.387779e-17`
  at `0.1` to `2.000000e+00` at `1e16` — eighteen orders of magnitude apart. A tolerance sized
  for values near `1.0` will reject perfectly correct results near `1e16`.
- **Assuming a broken cast just truncates toward zero.** IEEE-754's default rounding mode is
  round-to-nearest, ties-to-even, not truncation; a cast that only truncates introduces a
  one-sided bias on every halfway case.
  [This debugging task](../tasks/num-fix-truncating-cast-to-ties-to-even/task.md) gates
  `byte_exact_fraction == 1.0` against a reference that gets the tie-breaking bit right.

## Where else to practise this

Honest comparison, from the [full survey of what exists](../LANDSCAPE.md):

- **[Float Exposed](https://float.exposed/)** — type a decimal or flip individual bits of a
  half/bfloat16/float/double and see the exact base-10 value and the ULP gap to its neighbours
  live. The best hands-on companion to the ULP table above; nothing to submit or grade.
- **[fp-conv](https://sw23.github.io/fp-conv/)** — the same click-a-bit interaction extended to
  fp8/fp6/fp4 and custom bit layouts, past what this page's binary64 focus covers.
- **[array-api-tests](https://github.com/data-apis/array-api-tests)** — a conformance suite
  encoding NumPy's real dtype-promotion rules as executable assertions; not a tutorial, but the
  most rigorous runnable artifact found anywhere for that adjacent question.
- The survey's own verdict for this area is specific: IEEE-754 bit-level exploration is well
  served by two independent visualizers, but "nobody grades ... a stable-softmax/log-sum-exp
  implementation against an overflow-triggering input, scores detection of an in-place aliasing
  bug, [or] quizzes dtype-promotion predictions against real NumPy casting rules" — the
  reconstruct-from-bits and epsilon-derivation tasks linked above are this bank's answer for the
  representation half of that gap.

## References

1. IEEE Standard for Floating-Point Arithmetic (IEEE 754-2019).
   https://ieeexplore.ieee.org/document/8766229
2. Goldberg, D., *What Every Computer Scientist Should Know About Floating-Point Arithmetic*,
   ACM Computing Surveys, 23(1), 1991. https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
3. Python documentation, *15. Floating Point Arithmetic: Issues and Limitations*.
   https://docs.python.org/3/tutorial/floatingpoint.html
