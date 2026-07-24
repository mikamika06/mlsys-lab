## Context

A SIMD instruction packs several scalar operations into one: NEON's
`float32x4` does 4 multiply-adds per instruction, AVX2's 256-bit float lanes
do 8, AVX-512 does 16. If a dot product's length `n` were always an exact
multiple of the vector width `W`, the speedup over scalar code would simply
be `W` — every scalar instruction becomes one lane of one vector
instruction.

Real dot products rarely cooperate. When `n` isn't a multiple of `W`, the
last `n % W` elements don't fill a whole vector register, so a real
implementation falls back to plain scalar instructions for that leftover
"tail" — one instruction per leftover element, getting none of the
per-instruction parallelism the rest of the loop enjoyed. The *modeled*
speedup has to account for that:

$$
\text{vector\_instrs}(n, W) = \left\lfloor \frac{n}{W} \right\rfloor +
\left(n \bmod W\right)
$$
$$
\text{speedup}(n, W) = \frac{n}{\text{vector\_instrs}(n, W)}
$$

The first term is the number of full-width vector instructions (each one
replaces `W` scalar instructions); the second term is the leftover tail,
handled one scalar instruction at a time. When `n` is an exact multiple of
`W` the tail term is `0` and the speedup is exactly `W`. As the tail grows
relative to `n`, the speedup shrinks — and when `n < W`, every element is
tail, and the speedup collapses to exactly `1.0` (no vectorization benefit
at all: not even one full-width instruction was issued).

## Task

Implement, in `solve.cpp`:

```cpp
double modeled_vector_speedup(int n, int width);
```

Compute `vector_instrs = n / width + n % width` (integer division/modulo)
and return `(double)n / (double)vector_instrs`.

## Example

The driver (`main.cpp`, fixed) runs seven `(n, width)` pairs at widths
matching real SIMD ISAs:

```
neon_exact n=64 width=4 speedup=4.000000
neon_tail n=67 width=4 speedup=3.526316
avx2_exact n=64 width=8 speedup=8.000000
avx2_tail n=100 width=8 speedup=6.250000
avx512_exact n=256 width=16 speedup=16.000000
avx512_single n=1 width=16 speedup=1.000000
avx512_all_tail n=15 width=16 speedup=1.000000
```

`neon_exact` (`n=64, width=4`) divides evenly, so `speedup` is exactly
`4.0`. `avx512_all_tail` (`n=15, width=16`) never fills a single vector
register — 15 elements, width 16 — so `vector_instrs = 0 + 15 = 15` and the
speedup is `15/15 = 1.0`, identical to running fully scalar.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires every printed speedup to be within `1e-9` of the
reference (`main.cpp` + `ref.cpp`) (`max_abs_err <= 1e-9`). Returning the
naive `width` unconditionally (ignoring the tail entirely) matches the three
`*_exact` cases but is wrong on the four cases with a nonzero remainder;
computing `vector_instrs` as `ceil(n/width)` (treating the tail as one more
*full* vector instruction, not `n % width` separate scalar ones) also
matches the exact cases but disagrees with the reference everywhere the tail
is nonzero.
