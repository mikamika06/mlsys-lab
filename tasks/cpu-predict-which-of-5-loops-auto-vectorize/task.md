## Context

Modern compilers (Clang, GCC) automatically vectorize a loop with SIMD
instructions (NEON on Apple Silicon, AVX on x86) only when they can prove it
is safe: no loop-carried dependency other than a compiler-recognized
reduction idiom, no unresolved pointer aliasing, no data-dependent trip
count or index, and (for floating point) no reassociation that would change
the rounded result. This task does not use a rule table to decide the
answer -- it asks the real compiler.

`main.cpp` compiles 5 fixed loops into this very executable at `-O2`, runs
each once, then disassembles its OWN binary with `otool -tV` and scans each
loop's machine code for a real NEON vector-register suffix (`.4s`, `.2s`,
`.2d`, `.16b`, `.8h`). That scan is the ground truth -- not a simulation of
what a compiler "should" do.

## Task

Implement, in `solve.cpp`:

```cpp
bool predictLoop1();  // elementwise_add:       a[i] = b[i] + c[i]
bool predictLoop2();  // carried_dep:            a[i] = a[i-1] + b[i]
bool predictLoop3();  // plain_sum:              s += a[i]
bool predictLoop4();  // branch_free_select:     a[i] = b[i] > 0 ? b[i] : 0
bool predictLoop5();  // nonuniform_index:       a[i] = b[(i*i) % N]
```

Return `true` if you believe that loop autovectorizes at `-O2` on this
compiler, `false` if you believe it stays scalar. See `sol.hpp` for the
exact body of each loop.

## Example

The driver exercises all 5 loops, disassembles itself, and prints one line
per loop as `loopN <predicted> <actual>` (each a 0/1), e.g.:

```
loop1 1 1
loop2 0 0
loop3 0 0
loop4 1 1
loop5 0 0
```

## What the gate checks

`exact_match` requires the candidate's full stdout to equal the reference's
stdout byte-for-byte. The `actual` column comes from disassembling the
running binary itself, so it is identical for every correct or incorrect
`solve.cpp` -- only the `predicted` column can differ, and any wrong
prediction changes that loop's printed line. The obvious wrong approach --
assuming a plain `s += a[i]` reduction always vectorizes, or that a
non-uniform (quadratic) index is "just a gather" and vectorizes too -- fails
because the real compiler does neither at `-O2`: unsafe floating-point
reassociation blocks the first, and the non-constant stride blocks the
second.
