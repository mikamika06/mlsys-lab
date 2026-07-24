## Context

When a CUDA kernel is launched with $B$ blocks and the GPU has $S$ streaming
multiprocessors (SMs), each capable of hosting at most $C$ concurrent
blocks, the blocks cannot all run simultaneously. They execute in
**waves**: each wave fills up to $S \times C$ block slots, and any
remainder spills into a final partial wave.

The number of waves is

$$W = \left\lceil \frac{B}{S \times C} \right\rceil$$

and the **tail effect** measures how under-utilised the last wave is. If
$r = B \bmod (S \times C)$ is the remainder, the last wave contains

$$B_{\text{last}} = \begin{cases} S \times C & \text{if } r = 0 \\ r & \text{otherwise} \end{cases}$$

blocks, and the last-wave utilisation is

$$U_{\text{last}} = \frac{B_{\text{last}}}{S \times C}$$

When $U_{\text{last}} \ll 1$, most SMs sit idle in the final wave — this is
the tail effect, a key source of launch overhead in GPU workloads with many
small kernels.

## Task

Write a real CUDA-C kernel `wave_calc` in `solve.cu`:

```c
__global__ void wave_calc(float* out, int num_blocks, int num_sms, int blocks_per_sm);
```

It computes the number of execution waves $W$ and the last-wave utilisation
$U_{\text{last}}$ for a hypothetical launch of `num_blocks` blocks across
`num_sms` SMs, each holding at most `blocks_per_sm` concurrent blocks, and
stores:

- `out[0]` — number of waves $W$
- `out[1]` — last-wave utilisation $U_{\text{last}}$

The kernel is launched as a single block of a single thread
(`grid = 1, block = 1`), so `threadIdx.x == 0 && blockIdx.x == 0` for the
one thread that runs. Use `%` and integer `/` for the ceiling-division and
remainder arithmetic; multiply by `1.0` somewhere in the utilisation
computation to force a real (non-truncating) division for `U_last`.

## Example

For `num_blocks=10, num_sms=4, blocks_per_sm=2`: `capacity = 8`,
`W = ceil(10/8) = 2`, `r = 10 % 8 = 2`, `B_last = 2`,
`U_last = 2/8 = 0.25`. So `out[0] == 2.0` and `out[1] == 0.25`.

## What the gate checks

`check.py` parses `solve.cu` with the real CUDA-C frontend
(`arena.cuda_c.CudaProgram`), launches `wave_calc` on the software GPU
(`arena.cuda_sim.GPU`) for six `(num_blocks, num_sms, blocks_per_sm)`
configurations, and compares `out[0]`/`out[1]` against $W$ and
$U_{\text{last}}$ computed independently with plain Python arithmetic
(`max_abs_err <= 1e-9`). The starter kernel body is empty — it never writes
`out[0]`/`out[1]`, so `gmem` stays at its `-1.0` sentinel and every case
fails.
