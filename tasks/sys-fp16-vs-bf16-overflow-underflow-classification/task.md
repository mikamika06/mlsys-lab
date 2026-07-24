## Context

An IEEE-754-style binary floating point format with $e$ exponent bits and
$m$ mantissa bits has bias $\text{bias} = 2^{e-1}-1$. Its largest finite
magnitude and smallest positive **subnormal** magnitude are

$$
x_{\max} = (2 - 2^{-m})\cdot 2^{\,2^{e}-2-\text{bias}}, \qquad
x_{\min,\text{sub}} = 2^{\,1-\text{bias}}\cdot 2^{-m}.
$$

**fp16** uses $e=5, m=10$, giving $x_{\max}=65504$ and
$x_{\min,\text{sub}} \approx 5.96\times10^{-8}$. **bf16** uses $e=8,
m=7$ (the same exponent range as fp32, traded for less mantissa), giving
a much larger $x_{\max}\approx 3.39\times10^{38}$ but the same relative
precision loss. A real value $x$ storing into a format:

- **overflows** if $|x| > x_{\max}$ (rounds to $\pm\infty$),
- **underflows** if $x \neq 0$ and $|x| < x_{\min,\text{sub}}$ (flushes to
  $\pm 0$, the value is lost entirely),
- is otherwise representable (**ok**) — finite, possibly with rounding
  error, but neither blown up to infinity nor flushed to zero.

Because bf16 has fp32's huge exponent range, values that overflow fp16
routinely stay "ok" in bf16 — this is the core reason mixed-precision
training prefers bf16 over fp16 for activations/gradients that can spike
in magnitude.

## Task

Implement `classify_fp_value(x)`:

```python
def classify_fp_value(x: float) -> dict:
    ...
```

Return `{"fp16": <cls>, "bf16": <cls>}` where each `<cls>` is one of the
three strings `"overflow"`, `"underflow"`, `"ok"`, applying the
definitions above independently for each format.

## Example

```python
classify_fp_value(100000.0)
# {"fp16": "overflow", "bf16": "ok"}   -- 1e5 > 65504 but well under bf16's ~3.39e38

classify_fp_value(1e-10)
# {"fp16": "underflow", "bf16": "ok"}  -- below fp16's ~5.96e-8 floor, fine for bf16

classify_fp_value(0.0)
# {"fp16": "ok", "bf16": "ok"}
```

## What the gate checks

The grader evaluates your function on fixed boundary values (`0`, the
exact fp16/bf16 max-finite and min-subnormal magnitudes and their
negatives, values just past each boundary) plus 20 randomly generated
values spanning magnitudes from $10^{-46}$ to $10^{39}$
(`np.random.default_rng` seeded), and compares the returned dict to an
independent oracle: fp16 thresholds are read directly from
`np.finfo(np.float16)`, and bf16 thresholds are derived from the format
formula above with $e=8, m=7$. `exact_match` requires both the `"fp16"`
and `"bf16"` classification to match on every value — using fp16's
narrow range for both formats, swapping the overflow/underflow
conditions, or treating exactly-boundary values inconsistently will all
produce mismatches on the boundary cases.
