## Context

The roofline model bounds a kernel's attainable throughput by two lines: a
memory-bandwidth line and a peak-compute line. For a device with peak
compute `peak_flops` (FLOP/s) and peak memory bandwidth `peak_bw`
(bytes/s), a kernel with arithmetic intensity `ai` (FLOP processed per byte
moved) achieves

$$\text{attainable} = \min(\text{peak\_flops},\; ai \times \text{peak\_bw})$$

The two lines cross at the **ridge point**:

$$\text{ridge} = \frac{\text{peak\_flops}}{\text{peak\_bw}} \quad \text{(FLOP/byte)}$$

Below the ridge point, the memory-bandwidth term is smaller, so the kernel
is **memory-bound** — it cannot go faster without moving less data.
At or above the ridge point, the peak-compute term is smaller (or equal),
so the kernel is **compute-bound** — more bandwidth would not help.

## Task

Implement

```cpp
void classify_regimes(double peak_flops, double peak_bw, const double* ai, int n, int* out);
```

For each `i` in `[0, n)`, compute `ridge = peak_flops / peak_bw` and write
`out[i] = 1` if `ai[i] >= ridge` (compute-bound), else `out[i] = 0`
(memory-bound). The boundary case `ai[i] == ridge` counts as compute-bound.

## Example

`peak_flops = 16e12`, `peak_bw = 2e12` gives `ridge = 8.0`. An intensity of
`4.0` (e.g. a bandwidth-heavy elementwise op) is memory-bound (`0`); an
intensity of `16.0` (e.g. a well-tiled GEMM) is compute-bound (`1`); an
intensity of exactly `8.0` is compute-bound (`1`), by the `>=` rule.

## What the gate checks

`exact_match`: the driver prints the ridge point and the full classification
vector for 8 fixed arithmetic intensities straddling the ridge (including
values just below, just above, and exactly at it). Getting the comparison
direction or the boundary rule wrong flips at least one label and fails the
match; a starter that never writes `out[]` leaves the driver's `-1`
sentinels in place, which also fails outright.
