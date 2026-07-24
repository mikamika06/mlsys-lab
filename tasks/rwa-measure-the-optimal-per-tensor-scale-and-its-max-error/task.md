## Context

Production FP8 quantizers (TensorRT-LLM, Transformer Engine, vLLM, ...) pick
a per-tensor scale that maps the tensor's peak magnitude exactly onto the
target format's largest representable value:

$$
\text{scale} = \frac{\max_i |x_i|}{\text{FORMAT\_MAX}}, \qquad
\text{FORMAT\_MAX} = \begin{cases} 448 & \text{E4M3 (1 sign, 4 exp, 3 mantissa bits)} \\ 57344 & \text{E5M2 (1 sign, 5 exp, 2 mantissa bits)} \end{cases}
$$

Quantizing then dequantizing through this scale,
$\hat{x} = \text{scale}\cdot\operatorname{round\_to\_grid}(x / \text{scale})$,
always keeps the format's dynamic range fully utilized (no wasted headroom,
no overflow). Because both formats are rescaled onto the *same* peak, the
comparison between them reduces purely to **mantissa precision**: E4M3's
extra mantissa bit halves the local quantization step at every exponent, so
on ordinary ("well-scaled") data it reconstructs more accurately than E5M2
— every time.

## Task

Implement `optimal_scale_and_error`:

```python
def optimal_scale_and_error(x: np.ndarray, fmt: str) -> tuple[float, float]:
    ...
```

* `x` — a `float64` NumPy array of any shape.
* `fmt` — `"e4m3"` or `"e5m2"`.

Compute `scale = max(|x|) / FORMAT_MAX[fmt]` (if `max(|x|) == 0`, return
`(0.0, 0.0)`). Quantize `x / scale` to the nearest representable minifloat
grid value of `fmt` (clamping at `±FORMAT_MAX`), dequantize by multiplying
back by `scale`, and return

```
(scale, max_abs_dequant_error)
```

where `max_abs_dequant_error = max_i |dequant(x)_i - x_i|`.

Subnormals follow the standard minifloat rule: for exponent field `0` the
value is $\text{sign}\cdot 2^{e_{\min}}\cdot(\text{mantissa}/2^m)$ (no
implicit leading 1); otherwise it is
$\text{sign}\cdot 2^{(\text{exp}-\text{bias})}\cdot(1+\text{mantissa}/2^m)$,
with bias $7$ for E4M3 and $15$ for E5M2.

## Example

```python
import numpy as np

x = np.array([1.0, -3.5, 8.0])
scale, err = optimal_scale_and_error(x, "e4m3")
# amax = 8.0 -> scale = 8.0 / 448 ≈ 0.01786
# x/scale ≈ [56, -196, 448] -- 448 is EXACTLY E4M3's max, so it round-trips
# perfectly; the other two round to the nearest grid point and scale back.
```

## What the gate checks

- **scale_rel_err** — your reported `scale` must match
  `amax(|x|) / FORMAT_MAX[fmt]` (computed by the grader) to a relative error
  `<= 1e-6`, across 6 random tensors of varying shape, scale, and format.
- **error_rel_err** — your reported `max_abs_dequant_error` must match a
  from-scratch reference quantizer's error to a relative error `<= 1e-6` on
  the same 6 tensors.
- **order_ok** — on 4 additional *well-scaled* random tensors (no extreme
  outliers), calling your function with `fmt="e4m3"` must give a strictly
  smaller error than `fmt="e5m2"` — confirming you actually implemented the
  format-specific mantissa precision correctly rather than just returning
  plausible-looking numbers.

All three gates must pass.
