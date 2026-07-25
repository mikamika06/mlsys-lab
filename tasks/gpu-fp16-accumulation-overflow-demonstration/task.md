## Context

`fp16` can only represent finite values up to `65504`. That's not a
precision limit — it's a hard ceiling. Accumulate enough positive
values into an `fp16` running sum and, the moment the true total
crosses `65504`, the accumulator **saturates**: every value it can't
represent past its own maximum collapses to that maximum (or to
infinity, depending on the rounding mode), and every further addition
is silently absorbed into a number that stops growing. This is exactly
why real training loops accumulate in `fp32` even when the surrounding
tensors are `fp16`: not for extra precision, but because summing enough
values in a low-range format doesn't just lose accuracy — it stops
tracking the true value at all.

## Task

Implement, in `solve.cu`:

```cuda
__global__ void accumulate_clamped(const float* x, float* out, int n, float clamp_max);
```

A single thread sequentially sums `x[0]` through `x[n-1]` into `acc`
(starting at `0.0`). After **every** addition, if `acc` has exceeded
`clamp_max`, saturate it: set `acc = clamp_max`. Write the final `acc`
to `out[0]`.

## Example

`clamp_max = 100`, values `[60, 60, 60]`: after the first addition
`acc = 60` (no clamp). After the second, `acc = 120`, clamped to `100`.
After the third, `acc = 160`, clamped again to `100`. Final result:
`100` — nowhere near the true sum of `180`.

## What the gate checks

The grader launches `accumulate_clamped` on the same fixed 200-value
fixture (each value drawn from `[300, 700)`, true sum around `97376`)
twice: once with `clamp_max` set to fp16's max finite magnitude
(`65504.0`), once with an effectively unbounded `clamp_max`
(`10^{30}`, standing in for `fp32`, whose own ceiling is nowhere near
this fixture's total). It requires

$$
\mathrm{fp32\_err} \le 10^{-6} \quad\text{and}\quad \mathrm{fp16\_err} \ge 1000
$$

On this fixture, the unbounded run tracks the true sum to within
`3 \times 10^{-11}` — essentially exact — while the fp16-clamped run
saturates at exactly `65504.0`, off by **31872.15**: not a rounding
error, a ceiling the accumulator physically cannot cross.
