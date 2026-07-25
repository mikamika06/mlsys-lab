## Context

A row of 32 elements, one per lane of a warp, can compute a full,
numerically-safe softmax without ever touching shared memory: warp
shuffles let every lane read every other lane's register directly.
`__shfl_down_sync` (used for a plain reduction) only leaves the final
answer at lane 0 — fine if only one lane needs it, but softmax needs
*every* lane to know the row's max and sum to normalize its own element.
`__shfl_xor_sync` in a **butterfly** ladder (`delta = 16, 8, 4, 2, 1`,
XOR-ing the lane index instead of offsetting it) solves that: after all 5
steps, every lane holds the *same*, fully-reduced value — free
broadcast, no extra step.

## Task

Implement, in real CUDA-C:

```cuda
__global__ void warp_softmax(float* out, const float* x, int n);
```

`val = x[threadIdx.x]`. Run a 5-step `__shfl_xor_sync` ladder
(`delta = 16, 8, 4, 2, 1`) computing `m = max` over the warp: at each
step, `float got = __shfl_xor_sync(0xffffffff, m, delta); m =
fmaxf(m, got);`. Then compute `e = expf(val - m)`, and run a second
5-step `__shfl_xor_sync` ladder summing `e` across the warp into `s`
(`s = s + got` at each step). Write `out[threadIdx.x] = e / s`.

## Example

32 lanes holding `1.0` through `32.0`: after the max ladder, every lane
holds `m = 32.0` (not just the lane that started with it). After the
sum ladder (on `e = expf(val - 32)`), every lane holds the same total
`s = sum(expf(x - 32))`, and each lane's own `out` is
`expf(x[lane]-32)/s` — normalized correctly using values it never read
from shared memory or a second kernel launch.

## What the gate checks

`max_abs_err <= 1e-9` on two independent 32-lane rows (64 elements, one
block) against a numpy oracle. Using `__shfl_down_sync` instead of
`_xor_` (leaves `m`/`s` correct only at lane 0, wrong everywhere else),
running the max ladder over `e` instead of over `val` (computing the max
of the exponentials, not the logits), or skipping the max-subtraction
before exponentiating, all produce a wrong per-lane output.
