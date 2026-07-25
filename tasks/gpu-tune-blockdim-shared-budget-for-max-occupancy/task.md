## Context

An SM can only keep so many blocks resident at once, and FOUR separate
budgets each cap that number independently — whichever is tightest
wins:

- **Threads**: the SM has a hard max thread count.
- **Shared memory**: every resident block reserves its own
  `shared_bytes_per_thread * blockDim` bytes; the SM only has so much.
- **Registers**: every resident block reserves `regs_per_thread *
  blockDim` registers across the whole SM's register file.
- **Hardware max blocks**: SMs also cap the raw *number* of
  simultaneously resident blocks, regardless of how much of the other
  three budgets they'd use.

Occupancy is how much of the SM's thread capacity actually gets filled:
`(resident_blocks * blockDim) / max_threads_per_sm`. A `blockDim` that's
too small under-fills threads even with plenty of blocks resident; one
that's too large runs out of shared memory or registers before it can
launch enough blocks to fill the SM either.

## Task

Implement

```cpp
__global__ void compute_occupancy(float* out, const float* block_dims, float regs_per_thread,
                                   float shared_bytes_per_thread, float max_threads_per_sm,
                                   float max_blocks_per_sm, float max_regs_per_sm,
                                   float max_shared_per_sm, int num_candidates);
```

For every candidate `bd = block_dims[i]` (`i` in `[0, num_candidates)`):

$$
\text{blocks\_by\_threads} = \left\lfloor \frac{\text{max\_threads\_per\_sm}}{bd} \right\rfloor, \quad
\text{blocks\_by\_shared} = \left\lfloor \frac{\text{max\_shared\_per\_sm}}{\text{shared\_bytes\_per\_thread} \times bd} \right\rfloor
$$
$$
\text{blocks\_by\_regs} = \left\lfloor \frac{\text{max\_regs\_per\_sm}}{\text{regs\_per\_thread} \times bd} \right\rfloor
$$
$$
\text{actual\_blocks} = \min(\text{blocks\_by\_threads}, \text{blocks\_by\_shared}, \text{blocks\_by\_regs}, \text{max\_blocks\_per\_sm})
$$
$$
\text{out}[i] = \frac{\text{actual\_blocks} \times bd}{\text{max\_threads\_per\_sm}}
$$

Use `floorf()` for every floor-division — this CUDA-C subset has no
int/float cast operator, and every value read from `float*` memory is
already just a plain number.

## Example

`bd=256, regs_per_thread=32, shared_bytes_per_thread=64,
max_threads_per_sm=2048, max_blocks_per_sm=32, max_regs_per_sm=65536,
max_shared_per_sm=98304`: `blocks_by_threads = floor(2048/256) = 8`,
`blocks_by_shared = floor(98304/(64*256)) = floor(6.0) = 6`,
`blocks_by_regs = floor(65536/(32*256)) = 8`. The tightest is `6`
(shared memory), so `actual_blocks = min(8, 6, 8, 32) = 6`, and
`out[i] = (6*256)/2048 = 0.75`.

## What the gate checks

`check.py` parses `solve.cu` with the real CUDA-C frontend and runs it
on 6 candidate blockDims (`32, 64, 128, 256, 512, 1024`) under the fixed
resource limits above, then compares the candidate's OWN maximum
occupancy across those 6 values against a reference maximum computed
independently with numpy. It requires

$$
\mathrm{max\_abs\_err} \le 10^{-6}
$$

On this fixture the true best achievable occupancy is `0.75`, reached at
several blockDims (`64, 128, 256, 512`) — the gate only checks that the
computed values reach that same ceiling somewhere, not which specific
blockDim wins. Forgetting one of the four budgets (e.g. never checking
`blocks_by_shared`) overstates `actual_blocks` for large blockDims and
reports occupancy above the true achievable maximum on at least one
candidate.
