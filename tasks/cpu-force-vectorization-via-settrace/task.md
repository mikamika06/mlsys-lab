## Context

SIMD hardware (NEON on Apple Silicon, AVX on x86) does not execute one
float at a time: a single vector instruction loads `width` contiguous
lanes into one register and operates on all of them at once. A
"vectorized" loop issues $n/\text{width}$ instructions for $n$ elements;
a loop that is only vectorized in name -- e.g. still looping one scalar
at a time and just calling itself "SIMD" -- issues $n$ instructions and
gets none of the speedup, even though it can still produce the exact
right numbers.

Because you can't read a hardware instruction counter from portable
C++, this task uses an instrumented hook (`op_tick`, declared in
`sol.hpp`, defined in `main.cpp`) that stands in for "one vector
instruction was issued". Your kernel must call it exactly once per
`width`-wide chunk it processes -- not once per element.

## Task

Implement

```cpp
void fma_vectorized(const float* a, const float* b, const float* c,
                     float* out, int n, int width);
```

Compute `out[i] = a[i]*b[i] + c[i]` for every `i` in `[0, n)`. `n` is a
multiple of `width`. Process the arrays in contiguous chunks of `width`
elements, and call `op_tick()` **exactly once per chunk** (so
`n/width` calls total) -- never once per element and never zero times.

## Example

With `n=8`, `width=4`: process lanes `0..3` as one chunk (call
`op_tick()` once), then lanes `4..7` as a second chunk (call
`op_tick()` again). Total: 2 calls, not 8.

## What the gate checks

`main.cpp` runs with `n=16`, `width=4`, prints the checksum of `out[]`
and the value of `g_vector_ops` (the number of `op_tick()` calls). The
gate is `max_abs_err <= 1e-9` over every printed number, so it catches
two different failures at once:

- Wrong arithmetic changes the checksum.
- Calling `op_tick()` once per element instead of once per chunk prints
  `vector_ops=16` instead of the reference's `vector_ops=4` -- a
  difference of 12, far above the tolerance -- even if the checksum
  itself is perfectly correct. Correct numbers computed the slow,
  unvectorized way still fail the gate.
