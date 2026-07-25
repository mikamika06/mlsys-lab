## Context

A warp is 32 lanes executing in lockstep. `__shfl_up_sync(mask, v, delta)`
lets lane `l` read the value `v` held by lane `l - delta` in the *same*
instruction — no shared memory, no barrier, no round trip through SMEM.
That makes the Hillis-Steele scan almost free inside a warp:

$$
\text{for } \delta \in \{1, 2, 4, 8, 16\}: \quad
v_{\text{new}} = \begin{cases} v + \mathrm{shfl\_up}(v, \delta) & \text{lane} \ge \delta \\ v & \text{lane} < \delta \end{cases}
$$

After the five steps, every lane holds the inclusive prefix sum of its
own warp: lane $\ell$ holds $\sum_{k=0}^{\ell} \text{in}[k]$, where index
$0$ is the first lane of *that warp*, not of the whole grid.

## Task

Implement

```cpp
__global__ void warp_inclusive_scan(float* out, const float* in, int n);
```

For every thread, `lane = threadIdx.x % 32`. Read `val = in[threadIdx.x]`,
then run the 5-step ladder above: at each step, `__shfl_up_sync` must be
the WHOLE right-hand side of its own assignment (read it into its own
variable, e.g. `float n1 = __shfl_up_sync(0xffffffff, val, 1);`), then
conditionally add it to `val` in a **separate** statement, guarded by
`lane >= delta`. Finally `out[threadIdx.x] = val;`.

## Example

For lane `3` at `delta=4`: `lane - delta = -1`, out of range — the
source lane doesn't exist, and `__shfl_up_sync` (matching real hardware)
returns lane `3`'s own value back. Adding that in would double it, which
is exactly why the guard `lane >= delta` (here `3 >= 4` is false) must
skip the add for this lane at this step.

## What the gate checks

`check.py` parses `solve.cu` with the real CUDA-C frontend and runs it on
64 threads (**two** warps) of a fixed random input, comparing the output
against a reference that computes `cumsum` independently inside each
32-element half — so a scan that never resets at the warp boundary (e.g.
using a global running sum instead of the per-warp shuffle ladder) fails
on the second warp even if the first is correct. It requires

$$
\mathrm{max\_abs\_err} \le 10^{-6}
$$

Dropping the `lane >= delta` guard and adding the shuffled value in
unconditionally measures `max_abs_err ≈ 10.1` on this fixture — every
lane below `delta` doubles its own value instead of leaving it alone,
and that error compounds through the remaining steps of the ladder.
