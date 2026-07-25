## Context

A counted `for` loop performs **overhead operations** on every iteration that are
not part of the useful body. On a typical scalar pipeline these are:

- **Counter increment**: `i += 1` (one ALU operation)
- **Conditional branch**: `i < N ?` (one compare-and-branch)

That gives $K = 2$ overhead operations per iteration. For a loop that executes
$N$ iterations the total overhead is $K \cdot N = 2N$ operations.

**Loop unrolling** by a factor $U$ places $U$ copies of the body inside each
iteration, shrinking the iteration count from $N$ down to $\lfloor N/U \rfloor$
(integer division — any leftover elements would need a separate remainder
loop, whose own overhead this model ignores). The overhead drops to
$2 \lfloor N/U \rfloor$, and the number of overhead operations *eliminated* is

$$\Delta = 2 \!\left(N - \left\lfloor\frac{N}{U}\right\rfloor\right)$$

The data-element accesses of the body remain exactly $N$ either way; only the
per-iteration bookkeeping shrinks. Note $U = 1$ (no unrolling) must give
$\Delta = 0$.

## Task

Implement

```cpp
long long unroll_overhead_saved(long long N, long long U);
```

Return the exact number of overhead operations eliminated by unrolling an
`N`-iteration loop by factor `U`, using the $K = 2$ model above and
truncating (C++ `/`) integer division:

$$\Delta = 2 \left(N - N / U\right)$$

## Example

```
unroll_overhead_saved(8, 2)
# = 2 * (8 - 8/2) = 2 * (8 - 4) = 8

unroll_overhead_saved(50, 1)
# = 2 * (50 - 50/1) = 0        -- no unrolling, nothing eliminated

unroll_overhead_saved(17, 5)
# = 2 * (17 - 17/5) = 2 * (17 - 3) = 28
```

## What the gate checks

`main.cpp` calls your function on five fixed `(N, U)` pairs — including
`U == 1` (must yield `0`), `U == N` (fully unrolled), and pairs where `U`
does not evenly divide `N` (exercises truncating division) — and prints one
result per line. The grader compiles your `.cpp` with the real local
`clang++`, runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{every printed value matches the reference's exactly}
$$

A starter that ignores unrolling entirely (e.g. always reporting the
un-unrolled overhead $2N$) fails on every case, including the trivial
`U == 1` one, where the correct answer is $0$.
