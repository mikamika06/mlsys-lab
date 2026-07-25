## Context

The most common way to write an elementwise kernel is "one thread per
element": `i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) { ... }`.
This is only correct if the launch actually started at least `n` threads.
Real launch configurations are often chosen for occupancy — a number of
blocks that keeps the SMs busy — not recomputed from every `n` a kernel
might ever see, and a kernel tuned against a large `n` can quietly be
handed a `grid * block` far smaller than the data it's asked to process.

The **grid-stride loop** decouples the two entirely: instead of touching at
most one element, each thread starts at its own global index and then
jumps forward by the *total number of threads launched*
(`blockDim.x * gridDim.x`) each iteration, looping until it runs past `n`.
A huge grid finishes every thread's loop after one iteration — identical
behavior to the one-element-per-thread version. A tiny grid just makes
every thread loop more times. The amount of *work* a launch can correctly
cover stops depending on how many threads happened to be started.

## Task

Implement, in `solve.cu`, a kernel with this signature:

```cuda
__global__ void scale_grid_stride(float* out, const float* in, int n, float s);
```

Compute `out[i] = s * in[i]` for every `i` in `[0, n)`, correctly, for
**any** grid size — including one where `blockDim.x * gridDim.x < n`.
Start at `i = blockIdx.x*blockDim.x + threadIdx.x`, and loop
`for (; i < n; i = i + stride)` with `stride = blockDim.x * gridDim.x`.

## Example

The grader runs the same `n = 200`, `block = 32` kernel at three different
grid sizes: `grid = 7` (224 threads — enough to cover all 200 elements in
one pass), `grid = 2` (64 threads — each thread must loop about 3 times),
and `grid = 1` (32 threads — each thread must loop about 7 times). A
one-element-per-thread kernel (`if (i < n) { out[i] = s*in[i]; }`, no loop)
passes the first configuration by coincidence — it happens to have launched
enough threads — but at `grid = 2` and `grid = 1` it leaves most of the
array at its pre-launch sentinel value, off by more than `1000` in the
worst element.

## What the gate checks

`check.py` builds the fixture once, parses `solve.cu`, and runs
`scale_grid_stride` on the software GPU (`arena.cuda_sim.GPU`) at all three
grid sizes above, pre-filling `out` with a sentinel before each launch. It
requires `max_abs_err == 0.0`, taken as the worst error across **all
three** launches — a kernel correct only at the largest grid size still
fails the gate, which is exactly the point: a single grid-size test could
never have caught this bug.
