## Context

A roofline chart plots attained throughput against arithmetic intensity
(FLOP/byte). Every point sits on one of two lines: a rising
memory-bandwidth line (`attained = ai * peak_bw`) for low intensity, or a
flat compute-bound plateau (`attained = peak_flops`) for high intensity.
Given only scattered `(ai, attained)` measurements — the way a profiler
actually hands you data — you can recover both device parameters without
ever needing to know which line each point belongs to, *if* you have at
least one point deep in each regime: the lowest-intensity point is
(almost) certainly memory-bound, and the highest-intensity point is
(almost) certainly on the plateau.

## Task

Implement

```cpp
void fit_roofline(const double* ai, const double* attained, int n, double* peak_bw_out, double* peak_flops_out);
```

- Find the sample with the smallest `ai[i]`. It is memory-bound, so
  `*peak_bw_out = attained[i] / ai[i]` (the slope of the memory-bound
  line).
- Find the sample with the largest `ai[i]`. It sits on the plateau, so
  `*peak_flops_out = attained[i]` directly.

## Example

For a device with `peak_flops = 20e12` and `peak_bw = 2.5e12` (ridge
point `8.0`), the sample at `ai = 0.5` is deep in the memory-bound
region: `attained = 0.5 * 2.5e12 = 1.25e12`, so
`peak_bw_out = 1.25e12 / 0.5 = 2.5e12` — recovering the true bandwidth.
The sample at `ai = 50.0` is well past the ridge:
`attained = min(20e12, 50*2.5e12) = 20e12`, recovering the true peak
FLOP/s directly.

## What the gate checks

`max_abs_err` on the two fitted parameters printed by the driver: the 6
`(ai, attained)` samples are generated from the real roofline formula
(not hardcoded), with the extremes deliberately deep enough into each
regime that picking the wrong extreme index, or mixing up which output
gets the ratio vs. the raw value, produces a badly wrong fit. A starter
returning `0.0` for both fails outright.
