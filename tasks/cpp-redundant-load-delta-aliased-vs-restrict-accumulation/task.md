## Context

In C and C++, pointer aliasing prevents compilers from hoisting memory
loads out of a loop body. Consider accumulating an array `src` of length
`N` into a single destination `*dest`:

1. **Pessimistic aliased loop**: because `dest` might alias an element of
   `src`, the compiler must re-read `*dest` AND write it back on every
   single iteration: `L = 2N` loads (one from `*dest`, one from
   `src[i]`, each iteration), `S = N` stores.
2. **Optimized restrict/hoisted loop**: once the compiler can prove (or
   is told, via `restrict`) that `dest` cannot alias `src`, `*dest` is
   loaded into a register once before the loop, accumulated there, and
   stored back once after: `L = N + 1` loads, `S = 1` store.

This task makes those load/store counts real: every access to `*dest` or
a `src` element must go through a counted accessor, so the driver reports
what your code actually did, not a formula.

## Task

Implement, in `solve.cpp`, both loops declared in `sol.hpp`:

```cpp
void accumulate_aliased(double* dest, const double* src, int n);
void accumulate_hoisted(double* dest, const double* src, int n);
```

Use the **provided** `load_double(const double*)` / `store_double(double*,
double)` for every single access to `*dest` and every `src` element —
never dereference the raw pointers yourself.

`accumulate_aliased`: on each iteration `i`, load `*dest`, load
`src[i]`, and store their sum back into `*dest` — reload and re-store on
every iteration, exactly as a compiler that cannot rule out aliasing
must.

`accumulate_hoisted`: load `*dest` once before the loop into a local
accumulator, add every `src[i]` to that local (one load per element, no
loads/stores of `*dest` inside the loop), then store the final local
value into `*dest` once after the loop.

## Example

For `n = 100`: `accumulate_aliased` must produce exactly 200 loads
(100 `*dest` + 100 `src[i]`) and 100 stores. `accumulate_hoisted` must
produce exactly 101 loads (1 `*dest` + 100 `src[i]`) and 1 store. Both
must compute the same final numeric result — only the operation counts
differ.

## What the gate checks

The fixed driver (`main.cpp`) runs both loops over three fixed values of
`n` (10, 37, 100), resetting the load/store counters between calls, and
prints each run's load count, store count, the load/store deltas, and
both loops' final accumulated results. The gate is an exact string match
(`exact_match == 1.0`) against the reference's printed output: an extra
or missing reload, a store inside the hoisted loop, or a wrong
accumulated value all change the counts or the result and fail the gate.
