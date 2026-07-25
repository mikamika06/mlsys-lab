## Context

A CUDA compiler decides where every variable lives long before the
kernel runs. `__shared__` and `__constant__` declarations, and anything
pointing at global memory, have a fixed home regardless of register
pressure. Plain scalars are different: the compiler *wants* to keep them
in registers (fastest access, one per thread), but registers are a hard,
finite per-thread budget. Scalars are admitted to registers *in
declaration order* — the moment the running total of register-word
usage would exceed the budget, every later scalar candidate has nowhere
to go and **spills** to local memory (physically off-chip, cached but
far slower than a real register) instead.

Crucially, only variables that *actually became* registers count against
the budget — a spilled variable doesn't "use up" register space, since
it was never granted any.

## Task

Implement, in real CUDA-C:

```cuda
__global__ void classify_residency(float* label, float* spill, const float* kind,
                                    const float* size, float budget, int n);
```

Single-threaded (`threadIdx.x == 0` only). `kind[i]` is `0`
(register-candidate scalar), `1` (`__shared__`), `2` (global), or `3`
(`__constant__`); `size[i]` is its register-word size (only meaningful
for kind `0`). Walk `i = 0..n-1` with a running register-word total,
initially `0`: for `kind==0`, if `running + size[i] <= budget`, set
`label[i]=0`, `spill[i]=0`, and add `size[i]` to `running`; otherwise set
`label[i]=4` (spilled to local), `spill[i]=1` (and don't touch
`running`). For `kind` `1`/`2`/`3`, set `label[i]` to that same value and
`spill[i]=0` — never touching `running`.

## Example

`budget=8`, register-candidates of size `2, 3, 2, 4, 1, 5` arriving in
that order (with some shared/global/constant declarations interspersed,
which don't affect the budget at all): running totals `2, 5, 7`, then a
`4`-word candidate would push it to `11 > 8` — that one spills. The next
`1`-word candidate checks against the *still-7* running total (the
spilled one never counted): `7+1=8`, fits exactly, becomes a register.
The final `5`-word candidate then overflows `8+5=13 > 8` and spills too.

## What the gate checks

`max_abs_err <= 1e-9` on both `label[]` and `spill[]` for a fixed
10-variable listing (mixing all four kinds, `budget=8`), against a numpy
oracle running the identical rule. Letting a spilled variable's size
still count toward `running`, or admitting kind `1`/`2`/`3` variables
into the register count, changes at least one later variable's label.
