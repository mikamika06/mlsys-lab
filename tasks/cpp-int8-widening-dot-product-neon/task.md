## Context

In high-performance deep learning inference (e.g. quantized int8 LLM
kernels), dot products between 8-bit signed integer weights and
activations must be computed without intermediate overflow: `int8 * int8`
can already reach $\pm 16384$, and summing 16 or more of those overflows
a 16-bit accumulator.

ARM NEON (available on Apple Silicon) provides **widening** operations
built for exactly this: `vld1q_s8` loads 16 signed bytes into a 128-bit
register, `vmull_s8` multiplies 8 signed-byte lanes pairwise into an
`int16x8` result (widening *before* the product can overflow), and
`vpadalq_s16` pairwise-widens and accumulates that `int16x8` into a
32-bit `int32x4` accumulator. This is the real building block real int8
quantized inference kernels use.

## Task

Implement, in `solve.cpp`,

```cpp
std::vector<int32_t> int8_widening_dot_product(const std::vector<int8_t>& A,
                                                 const std::vector<int8_t>& B,
                                                 int M, int N);
```

`A` and `B` are `M x N` row-major int8 matrices (`N` is always a multiple
of 16 in the test). Return, for each row `m`,

$$\text{res}[m] = \sum_{n=0}^{N-1} A[m,n] \times B[m,n]$$

as an exact 32-bit signed integer.

Use `<arm_neon.h>` intrinsics to process 16 lanes at a time: load a
16-byte chunk of `A` and `B` with `vld1q_s8`, split each into low/high
8-lane halves with `vget_low_s8`/`vget_high_s8`, widening-multiply each
half with `vmull_s8` (`int8x8 x int8x8 -> int16x8`), and fold both halves
into a running `int32x4_t` accumulator with `vpadalq_s16`. Reduce the
4-lane accumulator to a scalar at the end of each row (e.g. with
`vadd_s32`/`vpadd_s32`/`vget_lane_s32`). A plain scalar loop over
`int32_t` would produce the same numbers but defeats the point of the
exercise — write the widening SIMD version.

## Example

For row `A = {1, 2, 3, 4, ...}`, `B = {5, 6, 7, 8, ...}` (padded to a
multiple of 16 with zeros for illustration), the dot product is
`1*5 + 2*6 + 3*7 + 4*8 + ... = 70 + ...`.

## What the gate checks

The fixed driver (`main.cpp`) runs four fixed `(M, N)` cases with
deterministic int8 data spanning the full `[-128, 127]` range, and prints
every row's dot product. The gate is an exact string match
(`exact_match == 1.0`) against the reference's printed output — integer
dot products are exact, so any overflow, truncation, or off-by-lane bug
in the widening/accumulation logic shows up as a wrong integer and fails
the gate immediately.
