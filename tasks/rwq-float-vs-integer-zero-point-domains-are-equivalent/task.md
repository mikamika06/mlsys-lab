## Context

Affine (zero-point) quantization maps a real tensor $x$ to integer codes $q$ with

$$
q = \mathrm{round}(x / s) + z, \qquad x \approx s \,(q - z),
$$

where $s$ is the scale and $z$ is the **zero point** — the integer code that represents
the real value $0$. Written this way, the zero point lives in the *integer domain*: it is
subtracted from $q$ before scaling.

Some quantization libraries instead store the zero point already folded into real-value
units, as an additive bias:

$$
\hat{x} = s\, q + b .
$$

Here $b$ (the *float-domain* zero point) plays the same role as $z$, but it is a real
number added after scaling rather than an integer subtracted before scaling.

The two forms must describe the same dequantized value for every code $q$:

$$
s\,(q - z) = s\,q + b \iff b = -s\,z \iff z = -\frac{b}{s}.
$$

So given $s$ and $b$, the equivalent integer-domain zero point is $z = -b/s$, and
dequantizing through either formula must produce identical numbers (up to floating-point
rounding).

## Task

Implement `dual_zero_point_dequant`:

```python
def dual_zero_point_dequant(codes: list[int], scale: float, zp_float: float):
    ...
```

- `codes`: integer array of quantized codes $q$.
- `scale`: positive float scale $s$.
- `zp_float`: the float-domain zero point $b$.

Return a 3-tuple `(deq_float_domain, deq_int_domain, zp_int)`:

- `deq_float_domain`: dequantize using the float-domain formula, $\hat{x} = s\,q + b$.
- `deq_int_domain`: derive the equivalent integer-domain zero point $z = -b/s$, then
  dequantize using $\hat{x} = s\,(q - z)$.
- `zp_int`: the derived integer-domain zero point $z$ itself (a Python float).

Both dequantized arrays must be `float64` list of the same shape as `codes`, and
must numerically agree with each other (they are the same real numbers computed two
ways).

## Example

```python

codes = [0, 5, 10, 20]
scale = 0.5
zp_float = -3.0

deq_f, deq_i, zp_int = dual_zero_point_dequant(codes, scale, zp_float)
# zp_int == -zp_float / scale == 6.0
# deq_f == deq_i == [-3.0, -0.5, 2.0, 7.0]
```

## What the gate checks

The gate builds a Python oracle with the same two formulas, computing the reference
`zp_int`, `deq_float_domain`, and `deq_int_domain` directly from `scale` and `zp_float`.
It checks:

- `max_abs_err`: the max absolute error of your two dequantized arrays against the
  oracle's two dequantized arrays.
- `domain_agreement_err`: the max absolute difference between your own
  `deq_float_domain` and `deq_int_domain` — the two formulas must agree with each other,
  not just with the oracle.
- `zp_int_err`: the absolute error of your derived `zp_int` against $-b/s$.

All three must be at most $10^{-6}$.
