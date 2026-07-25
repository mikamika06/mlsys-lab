## Context

A 2-stage elementwise chain — an affine transform followed by a ReLU —
can be run as two separate kernels:

$$\text{kernel 1: } t_i = a x_i + b \qquad \text{kernel 2: } y_i = \max(t_i, 0)$$

Each kernel is a full pass over global memory: kernel 1 reads $x$ and
writes $t$; kernel 2 reads $t$ back and writes $y$. Per element, that's
**4 global memory round-trips** total ($2$ reads $+$ $2$ writes), even
though $t_i$ is only ever needed for one instruction's worth of
arithmetic.

**Fusing** the two stages into ONE kernel — computing `v = a*x[i]+b` and
`y[i] = max(v, 0)` back to back, in registers, inside a single thread —
cuts that to **2 global round-trips per element** ($1$ read of $x$, $1$
write of $y$): the intermediate value $t_i$ never touches memory at all.
Half the memory traffic, for the exact same arithmetic.

## Task

Write a CUDA-C kernel:

```cpp
__global__ void fused_affine_relu(float* y, const float* x, float a, float b, int n);
```

One thread per element: `i = blockIdx.x * blockDim.x + threadIdx.x`.
Guard `i < n`, then compute `y[i] = relu(a * x[i] + b)` in one pass — read
`x[i]` once, write `y[i]` once, with the affine result kept only in a
local variable, never written back to `y` (or anywhere else in global
memory) as an intermediate step.

## Example

For $n = 256$ elements launched as 4 blocks of 64 threads (8 warps), each
warp does exactly 2 coalesced global-memory access steps — one read of
`x`, one write of `y` — for $8 \times 2 = 16$ transactions total. An
unfused, 2-kernel version of the same computation would need 4 access
steps per warp (read $x$, write $t$, read $t$, write $y$) — $8 \times 4 =
32$ transactions, exactly double, for identical results.

## What the gate checks

The grader parses your `.cu` with the CUDA-C frontend and runs it
thread-by-thread on the software GPU over a fixed random 256-element
fixture, checking the result against $\max(ax+b, 0)$ computed in numpy
(`max_abs_err <= 1e-9`) and the simulator's observed transaction count
(`transactions <= 16` — exactly what one coalesced read plus one coalesced
write per warp produces). Materializing the affine result to a second
global array before applying ReLU would double the transaction count and
fail the gate even with a perfectly correct final value. The empty
starter never writes `y` and fails the correctness gate.
