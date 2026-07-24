## Context

GPU kernel occupancy measures how many warps can be active on one compute unit at
the same time. A simplified occupancy model can be built from resource limits:
register usage, shared memory usage, and the hardware maximum number of warps.

For one program instance with $T$ threads and $W$ warps, the number of programs
that can be resident is limited by each resource:

$$
B_{\mathrm{reg}} =
\left\lfloor
\frac{R_{\mathrm{file}}}{r \cdot T}
\right\rfloor ,
$$

where $R_{\mathrm{file}}$ is the register capacity and $r$ is registers used per
thread.

Shared memory gives another limit:

$$
B_{\mathrm{smem}} =
\left\lfloor
\frac{S_{\mathrm{capacity}}}{S_{\mathrm{program}}}
\right\rfloor .
$$

The resident program count is

$$
B = \min(B_{\mathrm{reg}}, B_{\mathrm{smem}}, B_{\mathrm{max}}),
$$

and the active warp count is

$$
\mathrm{occupancy} =
\min(W_{\mathrm{max}}, B \cdot W).
$$

This task uses a simplified hardware model so the calculation can be tested
without a GPU.

## Task

Implement `predict_occupancy(regs_per_thread, smem_bytes, threads_per_program, limits)`.

The function arguments are:

- `regs_per_thread`: integer register usage per thread.
- `smem_bytes`: integer shared memory allocation per program.
- `threads_per_program`: integer number of threads in one program.
- `limits`: dictionary containing:
  - `warp_size`
  - `max_warps`
  - `max_blocks`
  - `register_file`
  - `smem_capacity`

Return the predicted number of active warps as an integer.

Assume all resource values are positive integers. The function should use integer
division for all resource limits.

## Example

```python
limits = {
    "warp_size": 32,
    "max_warps": 64,
    "max_blocks": 16,
    "register_file": 65536,
    "smem_capacity": 49152,
}

predict_occupancy(32, 4096, 128, limits)
# 64
```

## What the gate checks

The gate builds several hardware configurations and compares the submitted
implementation with an independent occupancy model computed inside the grader.

The returned value must exactly match the model for every tested configuration.
