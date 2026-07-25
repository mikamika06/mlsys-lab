## Context

**Thread coarsening**: instead of one thread computing one output, let it
compute $C$ outputs, reusing one shared load across all of them (register
blocking). Each extra output needs its own accumulator and address
bookkeeping, so registers-per-thread grow with $C$:

$$\text{regs}(C) = \text{base\_regs} + \text{regs\_per\_c} \cdot C$$

But every extra output is arithmetic performed on data already sitting in
a register — no new global load needed for the shared part — so
arithmetic intensity (FLOPs moved per byte of memory traffic) improves
with $C$ too:

$$\text{AI}(C) = \frac{\text{flops\_per\_elem} \cdot C}{\text{bytes\_per\_elem} \cdot (1 + C)}$$

This is monotonically increasing in $C$ (more reuse of the one shared
load, spread across more outputs) — so under a fixed register budget,
maximizing $\text{AI}(C)$ is exactly "take the largest $C$ the budget
allows":

$$C^* = \left\lfloor \frac{\text{reg\_budget} - \text{base\_regs}}{\text{regs\_per\_c}} \right\rfloor$$

## Task

Write a CUDA-C kernel (single thread — this derives two numbers from five
scalars):

```cpp
__global__ void coarsen_c(float* out, float reg_budget, float base_regs, float regs_per_c,
                            float flops_per_elem, float bytes_per_elem);
```

`out[0] = C*` (as derived above), `out[1] = AI(C*)`.

## Example

| reg_budget | base_regs | regs_per_c | flops_per_elem | bytes_per_elem | $C^*$ | AI($C^*$) |
|---|---|---|---|---|---|---|
| 40 | 6  | 2 | 2 | 4 | $\lfloor 17.0 \rfloor = 17$ | $34/72 \approx 0.472222$ |
| 64 | 8  | 4 | 2 | 4 | $\lfloor 14.0 \rfloor = 14$ | $28/60 \approx 0.466667$ |
| 32 | 10 | 1 | 4 | 4 | $\lfloor 22.0 \rfloor = 22$ | $88/92 \approx 0.956522$ |

## What the gate checks

The grader parses your `.cu` with the CUDA-C frontend and runs it (single
thread) on the software GPU once per fixed case, requiring `max_abs_err
<= 1e-6` against both `C*` and `AI(C*)` computed directly in Python.
Always coarsening by a fixed, "safe" `C = 1` never blows the register
budget, but it's off by `16`, `13`, and `21` on the three cases' `C*`
value alone — and its arithmetic intensity comes out at just `0.25` in
every case, nowhere near what the budget could actually buy.
