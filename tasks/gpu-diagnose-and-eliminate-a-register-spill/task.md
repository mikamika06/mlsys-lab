## Context

Every thread on a GPU gets a small, fixed budget of registers. When a
kernel needs to keep more values alive at once than that budget allows,
the compiler **spills**: instead of a register, the extra value lives
in "local memory" — which, despite the name, is physically just a
private slice of *global* memory. A spilled variable isn't a free
register access anymore; every read or write of it is a real
global-memory transaction, with real latency, competing with your
actual data traffic.

The fix is never "use less memory" in the abstract — it's reducing how
many values need to be **live at the same time**. Often that means
recomputing a cheap intermediate from a value you already have, instead
of computing it once and holding onto it (or, worse, writing it out and
reading it back) across a long stretch of the kernel.

## Task

`solve.cu` computes, for each `i < n`,

$$
\mathrm{out}[i] = \frac{(x[i]+1)(x[i]+2) - (x[i]+3)}{x[i]+4}
$$

but every intermediate term is round-tripped through a `scratch` buffer
in global memory instead of staying in a local variable — exactly what
a spill looks like once lowered: a store immediately followed, later,
by a load of the same value. The result is numerically correct, but
each of those round trips is real extra global-memory traffic.

**Fix it**: rewrite `compute_expr` so every intermediate value is an
ordinary local variable (`xv`, and one variable per arithmetic step)
and `scratch` is never read from or written to at all. The math must
stay identical — only where the intermediates live changes.

## Example

`x[i] = 2.0`: `(2+1)*(2+2) - (2+3) = 3*4 - 5 = 7`, divided by
`2+4 = 6`, giving `out[i] = 7/6 ≈ 1.1667`. Computing this with local
variables — `t1 = xv+1`, `t2 = xv+2`, `t3 = t1*t2`, `t4 = xv+3`,
`t5 = t3-t4`, `t6 = xv+4`, `out[i] = t5/t6` — needs zero traffic beyond
reading `x[i]` once and writing `out[i]` once.

## What the gate checks

The grader launches `compute_expr` over 128 threads (4 warps) against a
fixed input, checks the output against the closed-form expression
above, and reads the real transaction count from the simulator's own
coalescing model. It requires

$$
\mathrm{max\_abs\_err} \le 10^{-9} \quad\text{and}\quad \mathrm{transactions} \le 16
$$

The buggy version is already numerically perfect (`max_abs_err = 0`) —
this isn't a correctness bug, it's a memory-traffic bug. It measures
**148** transactions: one input read, one output write, and *six* extra
round trips per thread through `scratch` (4 stores + 2 loads), all
paid for in real global-memory segments. A version that keeps every
intermediate in a local variable measures **8** — just the essential
input and output traffic, over 18x less.
