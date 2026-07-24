## Context

A 32-bit `float` and a 32-bit unsigned integer occupy the same four bytes, but
they are different *types*. Sometimes you genuinely need to see one as the
other: to read a float's raw IEEE-754 bit pattern, or to build a float from a
known bit pattern. A single-precision float lays its 32 bits out as

$$\underbrace{s}_{\text{bit }31}\ \underbrace{e_7\dots e_0}_{\text{bits }30..23}\ \underbrace{m_{22}\dots m_0}_{\text{bits }22..0},\qquad
\text{value} = (-1)^{s}\,(1 + m/2^{23})\,2^{\,e-127}$$

for a normal number, where $e$ is the 8-bit *biased* exponent. Because the
exponent lives in a contiguous field, adding $k$ to $e$ multiplies the value by
$2^{k}$ exactly (as long as the result stays normal) — a classic bit hack.

There are two wrong ways to "convert" between a float and its bits, and one
right way:

* **Numeric cast** — `(uint32_t)x` rounds the *value* to an integer
  (`(uint32_t)1.5f == 1`). It discards the bit pattern entirely. This is a
  different operation, not a reinterpretation.
* **Pointer type-pun** — `*(uint32_t*)&x` reads the `float` object through an
  `unsigned*`. That is a **strict-aliasing violation**: undefined behavior.
  Under `-O2`, `clang++` assumes a `float*` and an `unsigned*` never refer to
  the same object, and may reorder, cache, or drop the access — so the code
  can silently compute the wrong answer even though it "looks right."
* **Byte copy** — `std::bit_cast<uint32_t>(x)` (C++20) or `std::memcpy` copies
  the four bytes and reinterprets them. It is well-defined and compiles to the
  same single move at `-O2`. This is the fix.

## Task

Implement in C++ (edit only `solve.cpp`):

```cpp
uint32_t float_to_bits(float x);              // raw IEEE-754 bits of x
float    bits_to_float(uint32_t b);           // float with bit pattern b
void     scale_pow2_inplace(float* x, int n, int k);  // multiply each by 2^k via the exponent field
```

`float_to_bits` and `bits_to_float` must reinterpret the bytes (use
`std::bit_cast` or `std::memcpy`), not perform a numeric conversion.
`scale_pow2_inplace` must add `k` to the biased exponent field of every element
through its bit pattern; the given inputs are normal and stay normal, and `k`
may be negative. The driver `main.cpp` and the contract `sol.hpp` are fixed.

## Example

```
float_to_bits(1.0f)        -> 1065353216   (0x3F800000)
float_to_bits(1.5f)        -> 1069547520   (0x3FC00000)   // NOT (uint32_t)1.5f == 1
bits_to_float(0x3F800000)  -> 1.000000                    // NOT (float)1065353216
scale_pow2_inplace({1.0f, 100.0f}, 2, 4) -> {16.0f, 1600.0f}   // x 2^4
scale_pow2_inplace({8.0f}, 1, -3)        -> {1.0f}             // x 2^-3
```

## What the gate checks

The fixed driver prints eight bit patterns, six reconstructed floats, and two
scaled arrays plus their sums. Your program's full output must match the
reference byte-for-byte (`exact_match == 1.0`). The starter compiles but uses
numeric casts and a no-op scale, so it prints different numbers and fails until
you reinterpret the bytes correctly.
