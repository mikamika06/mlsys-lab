## Context

A **fused multiply-add** computes $a \cdot b + c$ with a single rounding
step: the exact mathematical product $a \cdot b$ (computed as if with
infinite precision) is added to $c$, and only that final sum is rounded
to the target type:

$$
\mathrm{fma}(a,b,c) = \mathrm{round}(a \cdot b + c)
$$

The "obvious" way of writing $a \cdot b + c$ in float arithmetic instead
performs **two** roundings -- the product is rounded to a float first,
*then* the addition is rounded again:

$$
\mathrm{naive}(a,b,c) = \mathrm{round}\big(\mathrm{round}(a \cdot b) + c\big)
$$

These two expressions are mathematically the same formula, but they are
not always the same *float*. The first rounding of the product can throw
away bits that would have mattered to the final result -- most visibly
when $a \cdot b$ and $c$ are close in magnitude and nearly cancel, so the
final result is small and every bit of the intermediate product's error
gets amplified relative to it. `std::fma` is defined by the C++ standard
to always compute the single-rounding form, using extra internal
precision the hardware provides for exactly this purpose.

## Task

Implement:

```cpp
float fma_result(float a, float b, float c);   // single rounding
float naive_result(float a, float b, float c);  // two roundings
```

`fma_result` must compute the single-rounded value -- use `std::fma`.

`naive_result` must compute the double-rounded value: round `a * b` to a
float, then add `c` and round again. Store the intermediate product in a
`volatile float` before adding `c`. Without that, the compiler is
allowed (within one unbroken expression such as `return a * b + c;`) to
silently fuse the multiply and add back into a single hardware FMA
instruction -- which would make `naive_result` compute the *single*-
rounded value too, erasing the difference this task is built to show.

## Example

For `a = 1.29811692f, b = 1.39331698f, c = -1.80703771f`:

- The exact product $a \cdot b \approx 1.808\,706\ldots$; added to $c$,
  the true mathematical sum is a small positive number.
- `fma_result` rounds that once: `0.00165064423`.
- `naive_result` first rounds $a \cdot b$ to the nearest float (losing
  the low bits that mattered for this near-cancellation), then adds `c`
  and rounds again: `0.00165069103` -- a different float, off by about
  `4.7e-8`, roughly 100 ULP at this magnitude.

Not every input shows a difference: for `a = 2.0f, b = 3.0f, c = 4.0f`,
both the product and the sum are exactly representable in float, so
there is nothing to round away and both functions return `10` exactly.

## What the gate checks

`main.cpp` runs six fixed `(a, b, c)` triples -- five engineered for
near-cancellation, one with exact integer inputs -- through both
functions and prints `fma`, `naive`, and their difference for each. The
gate is `max_abs_err <= 1e-9` against the reference's printed numbers.
Computing `naive_result` the same way as `fma_result` (e.g. writing it
as the single expression `return a * b + c;`, which the compiler is
allowed to fuse into one hardware FMA instruction) makes every
difference collapse to `0`, which is off by up to `~7.7e-8` on the
near-cancellation cases and fails the gate.
