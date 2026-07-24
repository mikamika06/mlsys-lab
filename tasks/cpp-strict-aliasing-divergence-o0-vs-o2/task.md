## Context

The C++ **strict-aliasing rule** says a glvalue of type `T` may only access an
object whose dynamic type is (roughly) `T`, a signed/unsigned variant of `T`, or
`char`/`std::byte`. `float` and `uint32_t` are *unrelated* types, so reading the
bits of a `float` through a `uint32_t*` is **undefined behavior**:

```cpp
float f = 3.14f;
uint32_t u = *(uint32_t*)&f;   // UB: reads a float object through a uint32_t glvalue
```

At `-O0` the compiler emits the naive load and this *appears* to work. At `-O2`
the optimizer is allowed to assume the two pointers never alias, so it may
reorder, cache, or drop stores/loads — producing **different output at `-O2`
than at `-O0` from the same source**. That divergence is the classic symptom of
a strict-aliasing violation.

The fix is to copy the bytes through a permitted channel. Any of these is
well-defined and compiles to the same single load/store at `-O2`:

- `std::bit_cast<uint32_t>(f)` (C++20), `std::bit_cast<float>(u)`
- `std::memcpy(&u, &f, sizeof u);`
- reading through `char*` / `std::byte*` and reassembling

This task exercises the safe idiom on a real compression primitive:
**mantissa-truncation quantization**. An IEEE-754 `float32` is
$(-1)^{s}\cdot 1.m \cdot 2^{e-127}$ with a 1-bit sign, 8-bit exponent, and
23-bit mantissa $m$. Dropping the low mantissa bits is exactly how reduced-
precision float formats shrink weights (e.g. `bfloat16` keeps only the top 7
mantissa bits).

## Task

Implement the two functions declared in `sol.hpp`:

- `void quantize_mantissa(float* x, int n, int keep_bits)` — for each element,
  keep the top `keep_bits` (clamped to `0..23`) bits of the 23-bit mantissa and
  zero the low `23 - keep_bits` bits, preserving sign and exponent. In place.
  Concretely, mask the bit pattern with
  $\text{mask} = \lnot\,(2^{\,23-\text{keep\_bits}} - 1)$.
- `int count_bits_lost(const float* x, int n, int keep_bits)` — return the total
  number of mantissa bits discarded across `x[0..n)`: the popcount of the low
  `23 - keep_bits` mantissa bits, summed over every element.

Reinterpret `float` <-> `uint32_t` with a **strict-aliasing-safe** mechanism
(`std::bit_cast` or `std::memcpy`). A raw pointer type-pun such as
`*(uint32_t*)&x[i]` is undefined behavior and may be miscompiled at `-O2`.

## Example

With `keep_bits = 10`, `quantize_mantissa` clears the low 13 mantissa bits of
each float. The driver quantizes a fixed 8-element array, prints each result to
6 decimals, its running sum, and `count_bits_lost` over the original values:

```
3.140625 2.718750 ... 65.375000
sum=...
lost=...
```

The starter leaves the array untouched and returns `0`, so it prints the
original (un-quantized) values and `lost=0`.

## What the gate checks

`main.cpp` is compiled together with your `solve.cpp` using
`clang++ -O2 -std=c++20`, run, and its printed numbers are compared to the
reference. The gate is `max_abs_err <= 1e-6` over every printed value (the eight
quantized floats, the sum, and the lost-bit count). A correct strict-aliasing-
safe implementation reproduces the reference bit-for-bit; an empty or
type-punning-broken one does not.
