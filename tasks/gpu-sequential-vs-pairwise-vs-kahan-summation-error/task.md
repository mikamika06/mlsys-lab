## Context

Floating point addition rounds to the nearest representable value. Add a
small number to a MUCH larger running total, and if the small number is
below the larger value's precision granularity (its ULP — unit in the
last place), the addition has **no effect at all**: the bits that would
have represented it get rounded away completely, silently, every single
time.

**Kahan (compensated) summation** fixes this without any extra memory
beyond one running compensation term `c`: on each addition, `c` captures
exactly the low-order bits that got rounded away, and the NEXT term gets
corrected by subtracting `c` before it's added — recovering, over many
additions, almost all of the precision a naive running sum throws away.

$$y = x_i - c \qquad t = s + y \qquad c = (t - s) - y \qquad s = t$$

## Task

Write a CUDA-C kernel (single thread — summation order is inherently
sequential here):

```cpp
__global__ void kahan_sum(float* out, const float* values, int n);
```

Implement the Kahan recurrence above over `values[0..n)`, writing the
final compensated sum to `out[0]`.

## Example

The fixture is deliberately adversarial: one huge value, a thousand
`1.0`s, then the huge value negated back off —
`[1e16, 1.0 (x1000), -1e16]`. The exact mathematical sum is `1000.0`.

A **naive running sum** (`s += values[i]`, no compensation) adds `1e16`
first, then tries to add `1.0` a thousand times — but `1e16`'s
floating-point granularity is already coarser than `1.0`, so every one of
those additions is silently swallowed and changes nothing. Subtracting
`1e16` back off at the end leaves:

```
naive sum:  0.0     (relative error: 100%)
```

**Kahan summation** on the exact same fixture:

```
kahan sum:  1000.0  (relative error: 0%)
```

This works even though this simulator's arithmetic actually runs at full
double precision under the hood (there's no real `fp32` truncation to
model here) — the fixture's magnitude ratio (`1e16` against `1.0`) is
extreme enough to trigger the identical catastrophic-cancellation failure
at double precision that a naive `fp32` sum shows on real hardware at a
much smaller ratio.

## What the gate checks

The grader parses your `.cu` with the CUDA-C frontend and runs it (single
thread) on the software GPU over the fixed 1002-element fixture, requiring
`rel_err <= 1e-9` against the true sum of `1000.0`. A naive running total
gets exactly `0.0` — 100% relative error — and fails outright; only a
genuine compensated accumulation recovers the correct answer. The empty
starter leaves `out[0]` at its `-1.0` sentinel.
