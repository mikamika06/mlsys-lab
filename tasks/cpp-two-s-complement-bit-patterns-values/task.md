## Context

A fixed-width integer is just a bag of bits; its *value* depends on how you agree
to read them. For an unsigned field of width $w$, the pattern $b_{w-1}\dots b_1 b_0$
denotes

$$\sum_{i=0}^{w-1} b_i \, 2^{i}.$$

Under **two's complement** the top bit is re-weighted to be negative, so the same
pattern denotes

$$-\,b_{w-1}\,2^{\,w-1} + \sum_{i=0}^{w-2} b_i \, 2^{i}.$$

A handy equivalent rule: mask the pattern to $w$ bits to get an unsigned value $u$;
if the sign bit ($u \,\&\, 2^{w-1}$) is set, the signed value is $u - 2^{w}$,
otherwise it is $u$. This is why an all-ones byte `0xFF` is $255$ unsigned but
$-1$ as a signed 8-bit value, and why `0x80` is the most negative 8-bit value
$-128$.

## Task

Implement in C++:

```cpp
long long twos_complement_value(unsigned long long bits, int width, int is_signed);
```

Interpret the low `width` bits of `bits` as an integer of that width. If
`is_signed` is nonzero, decode it as a two's-complement value; otherwise decode
it as an unsigned magnitude. Ignore any bits at or above position `width`.
`width` is between 1 and 32.

Edit only `solve.cpp`. The driver `main.cpp` and the contract `sol.hpp` are fixed.

## Example

```
twos_complement_value(0xFF, 8, 1)   ->   -1
twos_complement_value(0xFF, 8, 0)   ->  255
twos_complement_value(0x80, 8, 1)   -> -128
twos_complement_value(0x8000, 16, 1)-> -32768
twos_complement_value(0xDEADBEEF, 32, 1) -> -559038737
```

## What the gate checks

The driver decodes 20 fixed (pattern, width, signedness) cases and prints each
decimal value plus their sum. Your program's full output must match the reference
byte-for-byte (`exact_match == 1.0`).
