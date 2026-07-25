## Context

A vectorized elementwise loop processes `WIDTH` elements per iteration
(one SIMD register's worth of lanes — 4 lanes of a 128-bit float vector
here). That only works cleanly when `n` is an exact multiple of `WIDTH`.
The instant it isn't, the main vector loop can only cover
`(n / WIDTH) * WIDTH` elements — the largest multiple of `WIDTH` that fits
— leaving `n % WIDTH` elements at the end that never fill a whole vector
register. Real SIMD code handles this "tail" one of two ways: a scalar
loop over the remainder, or a masked vector operation that only writes the
valid lanes. Forget it, and the last `n % WIDTH` elements of the output
are simply never touched.

## Task

`sol.hpp` pins `WIDTH = 4`. Your starting point in `solve.cpp` is a
broken `vec_add`, modeled in plain scalar C++ (no actual SIMD intrinsics —
the bug is purely algorithmic):

```cpp
void vec_add(const float* a, const float* b, float* c, int n) {
    int main_loop_end = (n / WIDTH) * WIDTH;
    for (int i = 0; i < main_loop_end; i += WIDTH) {
        for (int lane = 0; lane < WIDTH; ++lane)
            c[i + lane] = a[i + lane] + b[i + lane];
    }
}
```

It runs the `WIDTH`-wide main loop correctly but never handles the tail.
Fix it: after the main loop, add a scalar loop that computes
`c[i] = a[i] + b[i]` for every remaining `i` up to `n`.

## Example

The driver (`main.cpp`, fixed) calls `vec_add` with $n = 22$
($22 \bmod 4 = 2$, so indices 20 and 21 form the tail), `a[i] = i`,
`b[i] = 0.5i`, and prints every element of `c`. The last few, correctly:

```
c[19]=28.500000
c[20]=30.000000
c[21]=31.500000
```

The broken starter's main loop only reaches `i = 20`
(`(22 / 4) * 4 = 20`), so `c[20]` and `c[21]` are never written and keep
their `-999.0` sentinel value:

```
c[19]=28.500000
c[20]=-999.000000
c[21]=-999.000000
```

## What the gate checks

The grader compiles `main.cpp` + your file with `clang++ -O2 -std=c++20`,
runs it, and requires every one of the 22 printed `c[i]` values to satisfy
`max_abs_err <= 1e-6` against the same driver linked with the reference.
Getting all 20 "full vector chunk" elements right is not enough — the 2
untouched tail elements sit at `-999.0` instead of their correct values,
a difference of roughly `1029`, which fails the gate immediately.
