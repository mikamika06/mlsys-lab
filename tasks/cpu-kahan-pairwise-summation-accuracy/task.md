## Context

Floating-point addition rounds every result to the nearest representable
value. When you add a small number to a much larger running total, the
small number can fall entirely below the total's ULP (unit in the last
place) and vanish — the addition executes, but the sum doesn't change at
all. Summing left-to-right, this happens over and over as the total grows,
and the lost bits are gone for good.

**Kahan summation** fixes this without switching to a wider type. It keeps
a second float, `c`, holding a running estimate of the error introduced by
the *previous* addition, and subtracts it off *before* the next one:

$$y = x_i - c \qquad t = s + y \qquad c = (t - s) - y \qquad s = t$$

`(t - s) - y` recovers, to full float32 precision, exactly how much of `y`
got rounded away when it was added to `s` — that's what gets fed back in
next time instead of being lost. The whole algorithm stays in float32
throughout; the accuracy gain comes purely from tracking the rounding
error explicitly, not from extra precision.

## Task

Implement

```cpp
float kahan_sum(const float* arr, int n);
```

exactly per the four-line update above (`sum` and `c` both start at
`0.0f`).

## Example

Summing `1e8f` followed by `100000` copies of `1.0f`: the true total is
`100100000`. A naive `sum += x` loop gets stuck at exactly `100000000.0` —
every single one of the 100000 increments is below the accumulator's ULP
(~8 at that magnitude) and is silently dropped. Kahan summation recovers
the exact true value, `100100000.0`.

## What the gate checks

`max_abs_err <= 1e-3` on two fixtures printed by the fixed driver: the
`1e8` + 100000×`1.0f` case above, and one million copies of `0.1f` summed
in sequence (where naive summation drifts to `100958.34` against a true
total of `100000.0`, purely from compounding rounding error over a million
additions). Kahan summation is a strictly sequential, data-dependent
computation, so any correct implementation reproduces the reference's
result bit-for-bit — the tight tolerance only screens out a genuinely
wrong (e.g. uncompensated, or naive-sum-in-disguise) implementation.
