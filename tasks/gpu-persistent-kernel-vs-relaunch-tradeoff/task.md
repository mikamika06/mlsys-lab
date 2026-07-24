## Context

Every GPU kernel launch incurs a fixed overhead $H$ (driver setup, grid
configuration, register allocation). For a short kernel the per-iteration
compute cost $C_{\text{iter}}$ may be comparable to $H$. Over $K$
iterations two strategies exist:

**Relaunch** — launch the kernel $K$ times, each doing one iteration:

$$C_{\text{relaunch}} = K\,(H + C_{\text{iter}})$$

**Persistent** — launch once; each thread loops $K$ times internally:

$$C_{\text{persist}} = H + K \cdot C_{\text{iter}}$$

The savings are $\Delta = C_{\text{relaunch}} - C_{\text{persist}} = (K-1)\,H$.
As $K$ grows, the persistent kernel amortises launch overhead, but it
occupies SM resources for the entire duration and cannot be interleaved
with other work. The trade-off is fundamental to GPU workload design — and
on the real software-GPU simulator here, it isn't just theory: a
persistent kernel really does touch global memory only once per thread,
while $K$ relaunches really do touch it $K$ times, so the simulator's own
transaction-driven cycle count really is higher for relaunching.

## Task

Write three real CUDA-C kernels in `solve.cu`:

```c
__global__ void persistent_kernel(float* gmem, int N, int K);
__global__ void relaunch_kernel(float* gmem, int N);
__global__ void model_launch_cycles_kernel(float* out, int launch_overhead, int compute_cost_per_iter, int K);
```

1. `persistent_kernel`: one thread per element (`i = blockIdx.x * blockDim.x
   + threadIdx.x`, guarded by `i < N`). Load `gmem[i]` once into a local,
   add `1.0` to it in a `for` loop that runs `K` times, then store the
   result back to `gmem[i]` once.

2. `relaunch_kernel`: same indexing/guard, but does exactly one increment —
   `gmem[i] = gmem[i] + 1.0;` — no loop. The host launches this kernel `K`
   times to do the same total work as one `persistent_kernel` launch.

3. `model_launch_cycles_kernel`: a single thread (`threadIdx.x == 0 &&
   blockIdx.x == 0`) computes both formulas above and stores
   `out[0] = launch_overhead + K * compute_cost_per_iter` (persistent) and
   `out[1] = K * (launch_overhead + compute_cost_per_iter)` (relaunch).

## Example

For `N=4, K=3`, starting from `gmem = [0,0,0,0]`: after `persistent_kernel`
(one launch), `gmem == [3,3,3,3]`. Calling `relaunch_kernel` three times in
a row from the same starting state produces the same final values, but the
simulator counts more total memory transactions doing it that way (one
load + one store *per launch*, instead of one load + one store total).

## What the gate checks

`check.py` parses `solve.cu` with the real CUDA-C frontend and:

1. Launches `persistent_kernel` once over 64 elements with `K=5` and
   compares the result to `init + 5` (`max_abs_err <= 1e-6`).
2. Launches `relaunch_kernel` 5 times in a row over the same 64 elements
   and sums the simulator's real `cycles` metric across all 5 launches,
   then checks that total is strictly greater than `persistent_kernel`'s
   single-launch `cycles` (`relaunch_slower == 1`).
3. Launches `model_launch_cycles_kernel` for three `(H, C, K)` triples and
   compares `out[0]`/`out[1]` against the formulas computed independently
   in Python (`model_correct <= 1e-9`).

The starter's three kernel bodies are all empty, so `gmem` never changes
(fails gate 1), no cycles are spent doing anything different between the
two strategies (fails gate 2), and `out` stays at its `-1.0` sentinel
(fails gate 3).
