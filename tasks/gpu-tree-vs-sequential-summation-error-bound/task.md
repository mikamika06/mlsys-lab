## Context

Floating-point addition rounds to the precision of the *larger* operand.
Add a tiny value to an already-large running total, and if the tiny
value is smaller than the total's own rounding granularity, it vanishes
completely — not approximately, *exactly* zero effect, forever. Summing
left to right, every tiny value in a mostly-tiny array gets added into
an ever-growing running total one at a time, so this loss compounds:
the error grows with $n$, roughly $O(n \cdot \varepsilon)$.

A **tree (pairwise) reduction** avoids this by combining values in pairs
of roughly comparable magnitude at every step: many small values first
combine with *other small values their own size*, growing into a
larger, more significant partial sum *before* ever being combined with
anything huge. Its error only grows with the reduction's *depth*,
$O(\log_2 n \cdot \varepsilon)$ — exponentially better for exactly this
kind of wide-dynamic-range data.

## Task

Implement

```cpp
__global__ void tree_sum(float* out, const float* x, int n);
```

for `n = 1024`, launched as one block of `1024` threads. Load
`sdata[tid] = x[tid]`, `__syncthreads()`, then for `stride = 512, 256,
128, ..., 1`: if `tid < stride`, `sdata[tid] = sdata[tid] +
sdata[tid + stride]`, followed by `__syncthreads()` on every iteration
(not just the active ones). Thread `0` writes `out[0] = sdata[0]`.

## Example

Fixture: `x[0] = 1.0`, every other element `x[1..1023] = 1e-15`. The
*mathematically* exact total is `1.0 + 1023 * 1e-15 ≈
1.000000000001023`. Summed left to right, each `1e-15` addition to a
running total already at `1.0` is roughly `1e-15 / 1.0`, right at the
edge of float64's own rounding granularity (`~2.22e-16`) — most of those
1023 additions contribute essentially nothing, and the final result
drifts measurably off the true value. A tree reduction instead builds
up sub-sums of many `1e-15`s *before* they ever meet the `1.0`, so by
the time that combination happens, the tiny-side partial sum is large
enough to register a real, correctly-rounded contribution.

## What the gate checks

`check.py` parses `solve.cu` with the real CUDA-C frontend, runs it on
the fixture above, and compares `out[0]` against a reference computed
with `math.fsum` — *exact* summation, immune to the rounding error this
task is about. It requires

$$
\mathrm{rel\_err} \le 10^{-14}
$$

On this fixture the tree reduction measures `rel_err = 0.0` exactly. A
plain left-to-right sequential accumulation of the same 1024 values
measures `rel_err ≈ 1.13 \times 10^{-13}` — more than 10x over the gate
— purely from the order values get combined in, with the exact same
arithmetic operations and the exact same inputs.
