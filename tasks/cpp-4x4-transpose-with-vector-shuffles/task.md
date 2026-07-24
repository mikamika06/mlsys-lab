## Context

Modern CPUs use SIMD (Single Instruction, Multiple Data) to process multiple elements at once. On ARM NEON, 128-bit registers hold four 32-bit floats via the `float32x4_t` type. A standard challenge in graphics and matrix math is transposing a $4 \times 4$ tile. Instead of loading and storing individual floats, you use vector shuffle instructions to transpose it register-to-register.

`<arm_neon.h>` provides the interleaving intrinsic used here:

- `vzipq_f32(a, b)`: interleaves two 4-lane vectors and returns a pair
  `float32x4x2_t` where `.val[0] = [a0, b0, a1, b1]` and
  `.val[1] = [a2, b2, a3, b3]`.

## Task

Implement `transpose4x4` in `solve.cpp`:

```cpp
void transpose4x4(const float* in, float* out);
```

`in` and `out` each point at 16 floats forming a row-major $4 \times 4$ tile
(row $r$ occupies `in[4r .. 4r+4)`). Write the transpose of `in` into `out`:
`out[4c + r] = in[4r + c]` for every $r, c \in [0,4)$.

Do it with real ARM NEON shuffles, not scalar index swapping: load the four
rows with `vld1q_f32`, compose `vzipq_f32` calls to interleave them into the
transposed rows, and store with `vst1q_f32`. Two stages are enough — zip rows
$(0,2)$ and $(1,3)$ first, then zip the two results together lane-by-lane.

The fixed driver in `main.cpp` runs your function on three deterministic $4
\times 4$ tiles (an arange tile and two fixed pseudo-random tiles) and prints
every output element.

## Example

```
in  = [ 0,  1,  2,  3,
        4,  5,  6,  7,
        8,  9, 10, 11,
       12, 13, 14, 15]

out = [ 0,  4,  8, 12,
        1,  5,  9, 13,
        2,  6, 10, 14,
        3,  7, 11, 15]
```

Trace: zipping rows 0 and 2 gives `([0,8,1,9],[2,10,3,11])`; zipping rows 1
and 3 gives `([4,12,5,13],[6,14,7,15])`. Zipping the two low halves gives
`([0,4,8,12],[1,5,9,13])` — the first two output rows — and zipping the two
high halves gives `([2,6,10,14],[3,7,11,15])` — the last two.

## What the gate checks

The grader compiles `main.cpp` + your `solve.cpp` with real
`clang++ -O2 -std=c++20` against real ARM NEON (`arm_neon.h` on Apple
Silicon) and compares stdout byte-for-byte against the reference build
(`exact_match == 1.0`). The starter leaves `out` untouched (the caller's
sentinel `-1.0` values), so it prints `-1.000000` sixteen times per fixture
and fails every case until you wire up the real transpose.
