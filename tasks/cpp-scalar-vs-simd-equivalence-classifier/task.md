## Context

Whether a vectorized ("SIMD-style") kernel produces **bit-identical** results to an equivalent scalar loop depends on operation associativity and rounding:

1. **Elementwise operations** (`out[i] = a*x[i] + y[i]`): every output element is independent, so processing them in any order -- one at a time or in blocks of 4 like real SIMD lanes -- gives exactly the same bits as a scalar loop.
2. **Floating-point reductions**: IEEE-754 addition is **non-associative** -- `(a + b) + c` is not always bit-identical to `a + (b + c)`. A real 4-lane vector reduction keeps 4 separate running sums and only combines them at the end, which is a genuinely different order of additions than a scalar loop's single running total, and generally lands on a different last bit.
3. **Integer reductions**: integer addition is associative modulo $2^{32}$, so the same lane-split reduction produces exactly the same bits as a sequential scalar sum regardless.
4. **Fused multiply-add**: a scalar `a*x[i] + y[i]` (two separate roundings: one for the multiply, one for the add) can differ from a genuinely fused `std::fma(a, x[i], y[i])` (one rounding) by up to 1 ULP.

## Task

Implement the four kernels declared in `sol.hpp`: `simdSaxpy` (elementwise), `simdFloatSum` and `simdIntSum` (both must use a genuine 4-lane-accumulator reduction, not a single running total), and `simdFma` (must use `std::fma`, not a separate multiply and add).

The shipped `solve.cpp` implements every kernel as the plain scalar algorithm -- a single accumulator instead of 4 lanes, and `a*x[i]+y[i]` instead of `std::fma`.

## Example

```cpp
float simdFloatSum(const float* x, int n) {
    float lanes[4] = {0, 0, 0, 0};
    for (int i = 0; i < n; i++) lanes[i % 4] += x[i];
    return (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
}
```

## What the gate checks

`main.cpp` runs a fixed, seeded-deterministic data set through each of your four kernels alongside its own scalar reference implementation, and compares the raw bytes with `memcmp` -- it does not ask you (or `ref.cpp`) to predict anything, it observes what actually happens. Elementwise and integer-sum should read `1` (bit-exact); float-sum and fma should read `0` (they genuinely diverge). Your printed flags are compared against `ref.cpp`, compiled and run the same way: `max_abs_err <= 1e-9`. Implementing the reductions as a single running total (skipping the real 4-lane reassociation) or the FMA kernel as a plain multiply-then-add makes those two lines wrongly read `1`.
