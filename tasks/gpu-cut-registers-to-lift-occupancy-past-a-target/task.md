## Context

Every thread's local variables live in the SM's register file, shared
across every thread the SM is running at once. More registers per thread
means fewer threads (and fewer *blocks*) can be resident simultaneously —
which means fewer warps available to hide memory latency behind. This is
**occupancy**: how much of the hardware's thread-scheduling capacity your
kernel actually uses.

A simple, deterministic model of it: with a `REG_FILE_PER_SM`-register
file and `BLOCK`-thread blocks, register pressure alone caps how many
blocks can be co-resident at
$\lfloor \text{REG\_FILE\_PER\_SM} / (\text{regs\_per\_thread} \times
\text{BLOCK}) \rfloor$ — and the hardware itself caps it further at
`MAX_BLOCKS_PER_SM_HW`. Occupancy is the smaller of the two, as a fraction
of the hardware max:

$$\text{occupancy} = \frac{\min(\lfloor \text{REG\_FILE\_PER\_SM} / (\text{regs} \times \text{BLOCK}) \rfloor,\; \text{MAX\_BLOCKS\_PER\_SM\_HW})}{\text{MAX\_BLOCKS\_PER\_SM\_HW}}$$

Every extra named local variable that only holds a value used once —
instead of being recomputed on the spot, or folded into a running
accumulator — inflates `regs_per_thread` for no benefit, and can tank
occupancy in large, discrete steps (integer division punishes crossing a
register-file boundary hard).

## Task

`sum_sq_dev` computes $\text{out}[i] = (x_i-a)^2 + (x_i-b)^2 + (x_i-c)^2 +
(x_i-d)^2$. Your starting point in `solve.cu` gets this arithmetic right
but names a separate local variable for every deviation (`d0..d3`) and
every squared term (`s0..s3`), plus a final `total` — 10 live locals for
what only ever needs an index and a running sum.

Fix it: **recompute** each `(x[i] - k) * (x[i] - k)` inline and accumulate
it directly into a single running total, instead of storing every
intermediate in its own named variable. The kernel body should end up
with only 2 local variables: the thread index and the accumulator.

## Example

The grader statically counts distinct local-variable declarations in your
kernel's parsed AST (`regs_per_thread`), and plugs it into the fixed model
above with `REG_FILE_PER_SM = 2048`, `BLOCK = 128`, `MAX_BLOCKS_PER_SM_HW =
8`:

- `regs_per_thread = 10` (the broken version): $\lfloor 2048 / (10 \times
  128) \rfloor = \lfloor 1.6 \rfloor = 1$ block $\Rightarrow$ occupancy $=
  1/8 = 0.125$.
- `regs_per_thread = 2` (accumulate, don't store): $\lfloor 2048 / (2
  \times 128) \rfloor = 8$ blocks, capped at the hardware's own `8`
  $\Rightarrow$ occupancy $= 8/8 = 1.0$.

Both versions compute the exact same `out[i]` values — only the register
count, and therefore the modeled occupancy, differs.

## What the gate checks

The grader parses your `.cu`, walks the real AST to count local-variable
declarations, runs the fixed occupancy model, AND executes the kernel on
the software GPU for correctness. It requires `max_abs_err <= 1e-9`
against the numpy reference AND `occupancy_ok == 1.0` (modeled occupancy
$\ge 0.75$). The broken starter gets every output value exactly right —
`max_abs_err = 0` — but its 10 live locals cap modeled occupancy at
`0.125`, well under the `0.75` target, so `occupancy_ok` fails even though
correctness alone would pass. Getting the arithmetic right is not
enough — the register count itself has to come down.
