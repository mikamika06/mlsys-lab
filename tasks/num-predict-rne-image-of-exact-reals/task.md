## Context

Every finite `float32` represents an exact rational number. Given an exact
real number $v = \text{num}/\text{den}$ (arbitrary-precision integers,
$\text{den} > 0$), round-to-nearest-even (RNE) asks: which `float32` is
closest to $v$, breaking exact ties toward the representable value with an
**even** mantissa?

A normal `float32` has the form $f = (-1)^s \cdot 1.m \cdot 2^{e-127}$ with a
23-bit mantissa $m$. For a target exponent $e$, the 24-bit significand
(implicit leading 1 included) that exactly equals $v \cdot 2^{23-e}$ is, in
general, not an integer — write it as

$$
v \cdot 2^{23-e} = q + \frac{r}{d}, \qquad 0 \le r < d,
$$

an exact integer division with remainder. RNE keeps $q$, or $q+1$, by
comparing the remainder to half the divisor:

$$
q' =
\begin{cases}
q + 1 & 2r > d \\
q + 1 & 2r = d \ \text{and } q \text{ is odd (round to even)} \\
q & \text{otherwise}
\end{cases}
$$

If $q'$ overflows 24 bits ($q' = 2^{24}$), it wraps to $2^{23}$ and $e$
increments by one — a mantissa-to-exponent carry.

**The trap**: computing $v$ as a `float64` first (`float(num) / float(den)`)
and then casting that `float64` to `float32` performs **two** roundings.
Near a `float32` tie point, the first rounding (to 53 bits) can itself round
in a way that erases the information needed for the second rounding (to 24
bits) to land on the *correct* value — a classic "double rounding" bug. The
only way to get RNE image of the *exact* rational right in every case is to
round directly from the exact `(num, den)` pair, entirely in integer
arithmetic.

## Task

Implement `rne_fp32_bits`:

```python
def rne_fp32_bits(pairs: list[tuple[int, int]]) -> list[int]:
    ...
```

- `pairs` is a list of `(num, den)` pairs of Python `int`s (arbitrary
  precision — they may be far larger than fits in a `float64` mantissa).
  `den > 0` always; `num` may be negative, zero, or positive.
- For each pair, compute the exact rational $v = \text{num}/\text{den}$ and
  determine which `float32` it rounds to under round-to-nearest-even, as
  defined above.
- Return a list of Python `int`s, the same length as `pairs`, each the
  32-bit unsigned bit pattern of the corresponding rounded `float32`.
- Every test input is constructed to land in `float32`'s normal exponent
  range, so you do not need to handle subnormals, zero-exponent edge cases,
  overflow to infinity, or NaN — but `num == 0` (exact zero) does occur and
  must map to the all-zero-mantissa, all-zero-exponent bit pattern for the
  given sign.
- You must not round through an intermediate `float64` (no
  `float(num) / float(den)`, no float(num / den)`) — some test cases
  are specifically constructed so that route gives the wrong answer.

## Example

```python
num = (2**24 + 3) * 2**36 - 1
den = 2**60
# num/den is exact-real just BELOW the float32 halfway point between
# 1 + 1/2**23 and 1 + 2/2**23, so it must round DOWN to 1 + 1/2**23.
# float(num)/float(den) rounds to 1.0000001788139343 first (a float64),
# which is exactly the halfway point again — rounding that a second time
# to float32 ties-to-even UP instead, giving the wrong answer.

bits = rne_fp32_bits([(num, den)])
# bits[0] == 0x3f800001   (== float32(1 + 1/2**23), the correct single-rounded result)
# NOT 0x3f800002 (what double-rounding through float64 would give)
```

## What the gate checks

The grader builds a mix of cases: thousands of small random `(num, den)`
pairs, hundreds of pairs with 60-200 bit random integers (values a
`float64` cannot represent exactly), several exact `float32`-level ties at
various exponents (including one engineered to overflow the mantissa and
carry into the exponent), and several pairs specifically engineered to
produce a *different* answer under naive double-rounding through `float64`
than under correct direct rounding.

For every pair the grader computes the reference bit pattern with an
independent, pure-integer RNE implementation (the same algorithm described
above, executed once in the grader, never calling your code or hardcoding
expected values) and compares it against your output. The gate requires the
fraction of matching bit patterns, `exact_match`, to be exactly `1.0` — a
single mismatch anywhere (including the double-rounding traps) fails the
gate.
