## Context

ARM NEON (available on Apple Silicon) provides 128-bit SIMD registers that
hold four 32-bit floats (`float32x4_t`). A dot-product kernel multiplies
corresponding elements and accumulates into a 4-lane vector, then
performs a **horizontal sum** at the end. Because SIMD lanes are
independent, the horizontal reduction requires explicit pairwise lane
additions — you cannot simply pull the four lanes out as scalars and add
them in a hot loop; the reduction itself must be vectorized too, done
once at the end:

$$
\text{step 1:}\quad s_{01} = v_0 + v_1,\quad s_{23} = v_2 + v_3
$$
$$
\text{step 2:}\quad \text{result} = s_{01} + s_{23}
$$

For better instruction-level parallelism, real kernels often keep two (or
four) independent `float32x4_t` accumulators running over interleaved
groups during the multiply-accumulate loop, then merge them into one
before the final horizontal reduction.

## Task

Implement, in `solve.cpp`,

```cpp
float neon_dot(const std::vector<float>& a, const std::vector<float>& b);
```

`a` and `b` have equal length `n`, `n % 4 == 0`. Compute
$a \cdot b = \sum_{i=0}^{n-1} a_i b_i$ using `<arm_neon.h>`:

1. Process elements in groups of 4: load each group with `vld1q_f32` and
   multiply-accumulate lane-wise into a `float32x4_t` accumulator
   (`vmulq_f32` + `vaddq_f32`, or the fused `vfmaq_f32`) — never collapse
   a group to a scalar inside the loop. You may use two independent
   `float32x4_t` accumulators over interleaved groups of 4 for
   instruction-level parallelism, but merge them (`vaddq_f32`) before
   reducing.
2. After the loop, perform an explicit pairwise horizontal reduction:
   combine lanes `(0,1)` and `(2,3)` with `vadd_f32` on the low/high
   halves (`vget_low_f32`/`vget_high_f32`), combine those two partial
   sums with a second add (e.g. `vpadd_f32`), and extract the scalar with
   `vget_lane_f32`.

## Example

`neon_dot([1,2,3,4], [5,6,7,8])`: element-wise products
`[5, 12, 21, 32]`, pairwise sums `17` and `53`, horizontal sum `70.0`.

For 8 elements (two groups of 4): `neon_dot([1]*4 + [2]*4, [3]*4 +
[4]*4)` — group 1 contributes `[3,3,3,3]`, group 2 contributes
`[8,8,8,8]`; merged accumulator `[11,11,11,11]`, horizontal sum `44.0`.

## What the gate checks

The fixed driver (`main.cpp`) runs six fixed-length cases (`4, 4, 16, 16,
256, 128`) with deterministic, bounded-magnitude data, and prints each
result to 6 decimal places. The gate is `max_abs_err <= 1e-2` between the
printed values and the reference's — loose enough to tolerate the tiny
floating-point reassociation differences a different (still correct)
reduction order can produce, tight enough that a wrong reduction, a
scalar-collapsed loop, or a missed group still fails it.
