## Context

**SAXPY** (Single-precision $a \cdot X + Y$) is a foundational BLAS Level-1 routine that shows up everywhere: linear-algebra kernels, neural-network forward/backward passes, and numerical simulation inner loops. It is memory-bound and trivially data-parallel, which makes it the canonical first kernel for learning explicit SIMD.

On ARM NEON (ARMv8-A / Apple Silicon) a 128-bit vector register holds **four** 32-bit floats — the `float32x4_t` type. Instead of touching one element per iteration, you process four lanes at once. The relevant intrinsics from `<arm_neon.h>` are:

- `vdupq_n_f32(a)` — broadcast the scalar $a$ into all 4 lanes.
- `vld1q_f32(ptr)` — load 4 contiguous floats into a vector register.
- `vmlaq_f32(acc, b, c)` — elementwise $acc + b \cdot c$ (use `vfmaq_f32` for the fused form).
- `vst1q_f32(ptr, v)` — store 4 floats back to contiguous memory.

The update is elementwise and in place:
$$y_i \leftarrow a \cdot x_i + y_i \qquad i \in \{0, \dots, n-1\}.$$

Because the op order matches the scalar reference (one multiply + one add per element), the vector result agrees with scalar to within a rounding ULP.

## Task

Implement the contract in `sol.hpp`:

```cpp
void saxpy_neon(float a, const float* x, float* y, int n);
```

It must overwrite `y` in place with $y_i = a \cdot x_i + y_i$ for every $i \in [0, n)$, using NEON f32x4 intrinsics to process four lanes per iteration. `n` is a multiple of 4 in the driver, but keep a scalar tail loop so the kernel is correct for any `n`.

## Example

```cpp
float a = 2.5f;
float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
float y[4] = {10.0f, 20.0f, 30.0f, 40.0f};

saxpy_neon(a, x, y, 4);
// y -> {12.5, 25.0, 37.5, 50.0}
```

The fixed driver (`main.cpp`) builds a deterministic length-16 input, calls `saxpy_neon`, then prints each resulting `y[i]` with `%.6f` followed by their sum.

## What the gate checks

The driver's printed numbers are compared against the reference implementation's numbers. The gate passes when

$$\max_i \lvert y_i^{\text{you}} - y_i^{\text{ref}} \rvert \le 10^{-6}.$$

A correct NEON SAXPY lands well inside this tolerance; the starter (which leaves `y` untouched) does not, so it fails the gate.
