## Context

Thread coarsening — having one thread compute several output elements
instead of launching one thread per element — amortizes per-thread
overhead and gives the compiler more independent work to overlap. It has a
cost, though: every value a thread needs to keep around at once has to live
somewhere, and a GPU thread's "somewhere" is a small, fixed-size register
file shared across every thread resident on the core. Coarsen too
aggressively by keeping *all* of a thread's temporaries alive
simultaneously — instead of processing its elements one at a time and
reusing the same handful of registers — and the compiler runs out of
registers to hand out. The values that don't fit get **spilled**: pushed
out to (much slower) local memory, silently, with no error and no crash —
just a program that got slower by doing exactly what its author asked.

This simulator has no real register allocator to observe that with — so
this task models it the honest way that IS available: by counting how many
distinct scalar temporaries the kernel's own source declares. A thread that
unrolls 8 elements into 8 separately named intermediate values needs all 8
(plus their squared results) alive across most of the function; a thread
that processes the same 8 elements through a loop with one reused
temporary never needs more than a handful of names at once, no matter how
many elements it coarsens over.

## Task

Fix the kernel in `solve.cu`:

```cuda
__global__ void coarsened_square(float* out, const float* in, int n, float c);
```

Each thread must still process **8 consecutive elements**
(`base = (blockIdx.x*blockDim.x + threadIdx.x) * 8` through `base + 7`),
computing `out[i] = in[i]*in[i] + c` for each. Do it with a `for` loop over
`k` in `[0, 8)` and a **single** reused scalar temporary for the loaded
value, instead of eight separately named ones.

## Example

Both the buggy starter and the fix compute the exact same `out[i] = in[i]^2
+ c` for every element — `max_abs_err = 0.0` either way. What differs is
`modeled_registers`, the count of distinct local variable names the parsed
kernel source declares:

```
solve.cu (unrolled, v0..v7 + r0..r7): modeled_registers = 17  -> spill = 1
ref.cu   (loop, single reused `v`):   modeled_registers = 3   -> spill = 0
```

(`base`, `k`, and `v` — three names — cover the loop version regardless of
how many elements it coarsens over; the unrolled version needs a new name
for every value it wants to keep live at once.)

## What the gate checks

`check.py` parses `solve.cu`, walks its AST to count distinct local
variable declarations (excluding parameters and `__shared__` arrays,
neither of which are "extra" registers from coarsening), and requires
`spill == 0` against a fixed budget of 10 — *and* runs the kernel on the
software GPU (`arena.cuda_sim.GPU`) with a 4-block, 32-thread launch,
requiring `max_abs_err <= 1e-9` against a numpy oracle. Both gates have to
pass: a "fix" that changes the arithmetic while shrinking the variable
count fails on correctness, and the unrolled starter — numerically
correct, `max_abs_err = 0.0` — fails purely on `spill`.
