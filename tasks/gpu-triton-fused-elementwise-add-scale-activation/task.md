## Context

`out = relu(scale * (a + b))` is three logically separate ops — add,
scale, activation. Written as three separate kernels (the way a naive
autograd graph might schedule it), the intermediate `a+b` and
`scale*(a+b)` results each round-trip through global memory: written by
one kernel, read back by the next. **Fusing** them into a single kernel
keeps every intermediate value in a register for the lifetime of one
thread's element — global memory only sees `a[i]`, `b[i]` read once each
and `out[i]` written once, no matter how many logical ops got fused
together.

## Task

Implement, in real CUDA-C:

```cuda
__global__ void fused_add_scale_relu(float* out, const float* a, const float* b, float scale, int n);
```

For `i = blockIdx.x*blockDim.x + threadIdx.x`, guarded by `i < n`:
`out[i] = fmaxf(scale * (a[i] + b[i]), 0.0f)` — computed in one pass, no
intermediate array.

## Example

`a[i]=1.0, b[i]=-3.0, scale=1.5`: `sum=-2.0`, `scaled=-3.0`,
`relu(-3.0)=0.0`. `a[i]=2.0, b[i]=1.0, scale=1.5`: `sum=3.0`,
`scaled=4.5`, `relu(4.5)=4.5` — unchanged, since it's already positive.

## What the gate checks

`max_abs_err <= 1e-9` against `max(scale*(a+b), 0)` (numpy), **and**
`transactions <= 10` on a fixed 64-element, 2-block launch (reference
measures `6`: one coalesced transaction per array per block). Splitting
the computation across separate kernel launches with an intermediate
global-memory buffer, or reading `a[i]`/`b[i]` more than once, still
computes the right values but adds real, measurable global-memory
traffic beyond the fused budget.
