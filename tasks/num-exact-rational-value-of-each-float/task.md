## Context

Every finite IEEE-754 binary64 (`float`) value is, by construction, an *exact* rational
number: there is no rounding hidden inside the bit pattern itself — the rounding
already happened when the value was created. The 64 bits split into a sign bit $s$,
an 11-bit biased exponent $e$, and a 52-bit stored mantissa $m$. For a **normal**
number ($e \neq 0$):
$$
x = (-1)^s \times \left(1 + \frac{m}{2^{52}}\right) \times 2^{e-1023}
$$
For a **subnormal** number ($e = 0$, which also covers $\pm 0$), there is no implicit
leading 1:
$$
x = (-1)^s \times \frac{m}{2^{52}} \times 2^{1-1023}
$$

## Task

Implement two functions using only integer/bit operations on the raw byte pattern —
no `Fraction(x)`, no `x.as_integer_ratio()`, no float division:

```python
def float_fields(x: float) -> tuple[int, int, int]:
    """Return (sign_bit, biased_exponent, stored_mantissa) of x's binary64 pattern."""
    ...

def exact_ratio(x: float) -> tuple[int, int]:
    """Return (numerator, denominator) — the EXACT rational value of x, in lowest
    terms, with a positive denominator."""
    ...
```

Use `struct.pack`/`struct.unpack` to get at the raw 64 bits, then reconstruct the
value as a pair of Python `int`s (arbitrary precision, so this is exact even for the
smallest subnormal or the largest finite double) and reduce with `math.gcd`.

## Example

```python
float_fields(0.5)   # -> (0, 1022, 0)      # 1.0 * 2**(1022-1023) = 0.5
exact_ratio(0.5)     # -> (1, 2)

float_fields(-2.0)  # -> (1, 1024, 0)
exact_ratio(-2.0)    # -> (-2, 1)

exact_ratio(5e-324)  # -> (1, 2**1074)      # smallest positive subnormal
```

## What the gate checks

The grader draws a fixed set of floats spanning zero, subnormals, tiny/huge
normals, and random values across the full exponent range. For each value it
computes two independent oracles from real CPython/`struct` machinery (never
hardcoded): the raw `(sign, exponent, mantissa)` triple, and the exact rational
value via `fractions.Fraction(x)`.

* `fields_exact_fraction` — fraction of inputs where `float_fields` matches the
  oracle triple exactly (gate: `== 1.0`).
* `ratio_exact_fraction` — fraction of inputs where `exact_ratio` matches the
  oracle's reduced `(numerator, denominator)` exactly (gate: `== 1.0`).
* `max_rel_err` — worst-case relative error between `Fraction(*exact_ratio(x))`
  and the oracle fraction, computed with exact big-integer arithmetic before a
  single final float conversion (gate: `< 1e-9`).
