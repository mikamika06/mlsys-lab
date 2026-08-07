## Context

Two 8-bit floating-point formats are standard for quantizing LLM
tensors: **E4M3** (1 sign, 4 exponent bits, 3 mantissa bits, bias 7, finite
max magnitude $448$) and **E5M2** (1 sign, 5 exponent bits, 2 mantissa
bits, bias 15, finite max magnitude $57344$). Neither format is
universally "better" — which one round-trips a given raw value more
accurately depends entirely on that value's magnitude:

- Inside E4M3's range, its extra mantissa bit gives it roughly $2\times$
  finer quantization steps than E5M2 at the same exponent — E4M3 usually
  wins.
- Once a raw magnitude exceeds E4M3's max of $448$, E4M3 saturates
  (clamps) — a huge absolute error — while E5M2 still represents it as a
  normal number. E5M2 wins.
- For very small magnitudes near the subnormal floor, E5M2's much smaller
  minimum subnormal step ($2^{-16}$ vs E4M3's $2^{-9}$) again gives it
  finer resolution over a surprisingly wide low-magnitude band — E5M2 wins
  there too.
- For magnitudes small enough that *both* formats round to exactly $0$,
  the round-trip error is $|v|$ for both formats — an exact tie.

## Task

Implement `classify_better_format`:

```python
def classify_better_format(values: list[float]) -> list[int]:
    ...
```

For every scalar $v$ in `values` (any shape), round-trip it through both
formats: quantize to the nearest representable grid value (no rescaling —
just clamp/saturate at each format's finite max magnitude if $|v|$ exceeds
it), then measure the absolute round-trip error
$e_{\text{fmt}} = |\text{dequant}_{\text{fmt}}(v) - v|$.

Return an integer array of the same shape as `values` where each entry is

$$
\text{label}(v) = \begin{cases} 0 & e_{\text{E4M3}} \le e_{\text{E5M2}} \quad (\text{includes exact ties}) \\ 1 & \text{otherwise} \end{cases}
$$

Subnormals follow the standard minifloat rule: for exponent field $0$ the
value is $\text{sign}\cdot 2^{e_{\min}}\cdot(\text{mantissa}/2^m)$; for
exponent field $\geq 1$ it is
$\text{sign}\cdot 2^{(\text{exp}-\text{bias})}\cdot(1+\text{mantissa}/2^m)$.

## Example

```python

values = [1.5, 500.0, 1e-6]
labels = classify_better_format(values)
# 1.5   is well inside both ranges -> E4M3's extra mantissa bit wins -> 0
# 500.0 exceeds E4M3's 448 max (saturates) but not E5M2's -> 1
# 1e-6  underflows to 0 in BOTH formats -> equal error -> tie -> 0
```

## What the gate checks

A single gate, **exact_match**, compares your label array against a
from-scratch reference (built directly from the IEEE-754-style
exponent/mantissa/bias definitions above, not any library shortcut) on a
fixed fixture of 90 values (`values.npy`) spanning typical magnitudes,
raw outliers beyond E4M3's range, magnitudes beyond both ranges, values
near the subnormal floor of both formats, and a few exact boundary values
(`448.0`, `57344.0`, `0.0`, ...). Every label must match exactly.
