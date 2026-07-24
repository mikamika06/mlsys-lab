## Context

In C and C++, pointer **aliasing** occurs when two pointers might refer to
overlapping memory. Unannotated pointers must be assumed to alias, which
blocks useful loop optimizations. Consider a SAXPY loop,
$y[i] \leftarrow a \cdot x[i] + y[i]$, where `a` is read through a scalar
pointer `a_ptr`:

1. **Unhoisted** (compiler cannot prove `a_ptr` doesn't alias `y`): it
   must reload `*a_ptr` on every iteration — `3N` total loads (`N` scalar
   reloads + `N` loads of `x[i]` + `N` loads of `y[i]`), `N` stores.
2. **Hoisted** (`__restrict__`-style: `a_ptr` proven not to alias `x` or
   `y`): `*a_ptr` is loaded once before the loop into a register/local and
   reused every iteration — `1 + 2N` total loads, `N` stores.

This task makes those load/store counts real: every access to `*a_ptr`
or an `x`/`y` element must go through a counted accessor.

## Task

Implement, in `solve.cpp`, both loops declared in `sol.hpp`:

```cpp
void saxpy_unhoisted(const float* a_ptr, const float* x, float* y, int n);
void saxpy_hoisted(const float* a_ptr, const float* x, float* y, int n);
```

Use the **provided** `load_f(const float*)` / `store_f(float*, float)`
for every single access to `*a_ptr` and every `x[i]`/`y[i]` — never
dereference the raw pointers yourself.

`saxpy_unhoisted`: on each iteration `i`, load `*a_ptr`, load `x[i]`,
load `y[i]`, and store `a*x[i] + y[i]` into `y[i]` — reload `*a_ptr` on
every iteration.

`saxpy_hoisted`: load `*a_ptr` once before the loop into a local scalar;
inside the loop, load `x[i]` and `y[i]`, store the SAXPY result into
`y[i]` — never load `*a_ptr` again inside the loop.

## Example

For `n = 100`: `saxpy_unhoisted` must produce exactly 300 loads
(100 `*a_ptr` + 100 `x[i]` + 100 `y[i]`) and 100 stores.
`saxpy_hoisted` must produce exactly 201 loads (1 `*a_ptr` + 100 `x[i]` +
100 `y[i]`) and 100 stores. Both must compute the identical numeric `y`
values — only the operation counts differ.

## What the gate checks

The fixed driver (`main.cpp`) runs both loops over three fixed values of
`n` (10, 37, 100), resetting the load/store counters between calls, and
prints each run's load/store counts plus the first and last computed `y`
values from both variants. The gate is an exact string match
(`exact_match == 1.0`) against the reference's printed output: an extra
or missing `a_ptr` reload, or a wrong SAXPY result, changes the counts or
values and fails the gate.
