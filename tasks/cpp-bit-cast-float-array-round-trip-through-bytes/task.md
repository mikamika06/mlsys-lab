## Context

`reinterpret_cast<unsigned char*>(&x)` followed by dereferencing those bytes
as something other than `x`'s own type is classic **type-punning
undefined behavior**: the compiler is allowed to assume you only access an
object through its own type (the *strict aliasing rule*), and optimizing
under that assumption can silently produce wrong results when you break it.

C++20's `std::bit_cast<To>(from)` gives a **defined** way to reinterpret the
bits of one object as another type of the same size: it requires both types
to be trivially copyable and the same size, and it behaves as if the bytes
were copied with `memcpy` — no aliasing violation, no UB, and (unlike a raw
`memcpy` into a `To` you then have to trust is valid) the result is
immediately usable as a real `To` value.

## Task

Implement two functions that move `float` values through a raw byte buffer
using `std::bit_cast`, little-endian, 4 bytes per float:

```cpp
void floats_to_bytes(const float* x, int n, unsigned char* out);
void bytes_to_floats(const unsigned char* in, int n, float* out);
```

1. `floats_to_bytes`: for each `x[i]`, convert it to its bit pattern with
   `std::bit_cast<std::uint32_t>(x[i])`, then write that 32-bit pattern into
   `out[4*i .. 4*i+3]` **least-significant byte first** (byte 0 = bits 0-7,
   byte 1 = bits 8-15, byte 2 = bits 16-23, byte 3 = bits 24-31).
2. `bytes_to_floats`: for each group of 4 bytes, reassemble the
   `std::uint32_t` bit pattern in the same little-endian order, then convert
   it back with `std::bit_cast<float>(bits)`.

The round trip must be **bit-exact**, including the sign bit of `-0.0f`
(which compares equal to `0.0f` arithmetically but has a different bit
pattern — a correct implementation still tells them apart).

## Example

For `x = {1.5f}`:

```
bits  = 0x3FC00000                     // IEEE-754 bit pattern of 1.5f
bytes = [0x00, 0x00, 0xC0, 0x3F]       // little-endian
bytes_to_floats(bytes) -> 1.5f (bit pattern 0x3FC00000, unchanged)
```

## What the gate checks

The driver serializes a fixed 8-float fixture (mixing signs, integral and
fractional magnitudes, and `+0.0f`/`-0.0f`), prints the 32 serialized bytes,
deserializes them, and prints the recovered bit pattern of each float (via
`memcpy` into a `uint32_t`, not `%f`, specifically so a `-0.0f` that silently
became `+0.0f` cannot hide). Every buffer is poisoned before use, so a
function that returns without writing anything prints poison instead of
coincidentally-correct data. The grader compiles `solve.cpp` with
`clang++ -O2 -std=c++20`, runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{every printed byte and every printed bit pattern matches the reference}
$$
