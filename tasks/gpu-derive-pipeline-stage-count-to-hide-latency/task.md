## Context

Software pipelining (`cp.async` double/multi-buffering) hides a tile
load's latency $L$ by issuing the load for tile $k{+}1$ while the compute
for tile $k$ (taking $C$ cycles) is still running. With $S$ pipeline
stages, at any moment $S-1$ loads are in flight ahead of the stage
currently computing — so the compute engine has $(S-1) \cdot C$ cycles of
work queued up to run *while* the oldest in-flight load finishes.

That's enough to fully hide the latency exactly when
$(S-1) \cdot C \ge L$, i.e. $S \ge L/C + 1$. Since $S$ must be a whole
number of stages, and any shortfall (even a fraction of a cycle short)
leaves the compute engine stalled waiting on a load, the minimum
sufficient stage count rounds *up*:

$$S = \left\lceil \frac{L}{C} \right\rceil + 1$$

## Task

Implement, in real CUDA-C:

```cuda
__global__ void pipeline_stages(float* out, const float* L, const float* C, int n);
```

For `i = blockIdx.x*blockDim.x + threadIdx.x`, guarded by `i < n`:
`out[i] = ceilf(L[i] / C[i]) + 1.0f`.

## Example

`L=250, C=100`: `L/C = 2.5`, `ceil(2.5) = 3`, so `S = 4`. Check: 3 loads
in flight ahead of the current stage give `3*100 = 300 >= 250` cycles of
compute to hide the load — enough. With `S=3` (only 2 loads in flight),
`2*100 = 200 < 250`: the load hasn't finished when the compute engine
runs out of queued work. `L=100, C=100` (exact division): `ceil(1.0) = 1`,
`S = 2` — double buffering exactly suffices, no fraction wasted.

## What the gate checks

`max_abs_err <= 1e-6` on 6 fixed `(L, C)` pairs, including one exact
division (`L=999, C=333`) and one where the load latency is much smaller
than compute (`L=1, C=1000`, still needs `S=2`: pipelining always needs
at least double buffering to overlap anything at all). Using `floor`
instead of `ceil`, or dropping the `+1`, undercounts stages and fails to
hide the latency in exactly the cases the fixed pairs are chosen to
expose.
