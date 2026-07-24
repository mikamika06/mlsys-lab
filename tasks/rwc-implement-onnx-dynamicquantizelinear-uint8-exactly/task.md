## Context

ONNX's `DynamicQuantizeLinear` operator computes an asymmetric uint8
quantization of a float tensor $x$ **on the fly**, deriving the scale and
zero point from $x$'s own min/max rather than taking them as separate
inputs. With $q_{\min}=0$, $q_{\max}=255$:

$$
x_{\min} = \min(0, \min(x)), \qquad x_{\max} = \max(0, \max(x)),
$$

$$
y_{\text{scale}} = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}} = \frac{x_{\max}-x_{\min}}{255} .
$$

Forcing $0$ into both the min and the max guarantees the real value $0$ is
always exactly representable — necessary for correctness elsewhere (e.g.
zero-padding). The zero point is the quantized code that $0.0$ maps to:

$$
y_{\text{zero\_point}} = \mathrm{saturate}_{[0,255]}\!\Big(\mathrm{round}\big(q_{\min} - x_{\min}/y_{\text{scale}}\big)\Big),
$$

and every element is quantized the same way, reusing that same zero point:

$$
y_i = \mathrm{saturate}_{[0,255]}\!\Big(\mathrm{round}(x_i / y_{\text{scale}}) + y_{\text{zero\_point}}\Big).
$$

`round` is round-half-to-even (banker's rounding — NumPy's default
`np.round`/`np.rint`), and `saturate` clips to the closed interval
$[0,255]$ before casting to `uint8`.

## Task

Implement `dynamic_quantize_linear`:

```python
def dynamic_quantize_linear(x: np.ndarray) -> dict:
    ...
```

- `x` — a float array of any shape.

Return a `dict` with:

- `"y"` — `uint8` array, same shape as `x`: the quantized codes.
- `"y_scale"` — Python `float`.
- `"y_zero_point"` — `uint8` (or a value that compares equal to one), the
  scalar zero point.

## Example

```python
import numpy as np
dynamic_quantize_linear(np.array([-1.0, 0.0, 1.0, 2.0]))
# {"y": array([  0,  85, 170, 255], dtype=uint8),
#  "y_scale": 0.011764705882352941,
#  "y_zero_point": 85}
```

$x_{\min}=-1$, $x_{\max}=2$ (both already straddle $0$), so
$y_{\text{scale}} = 3/255$ and $y_{\text{zero\_point}} =
\mathrm{round}(1 / (3/255)) = \mathrm{round}(85.0) = 85$.

## What the gate checks

The grader runs your implementation on ten fixed activation tensors
(all-positive, all-negative, mixed-sign, tiny-magnitude, an
outlier-dominated tensor, and several random ones — every one with genuine
spread, so `y_scale` never degenerates to zero) and compares `y_scale`,
`y_zero_point`, and **every** element of `y` bit-exactly against the ONNX
spec's own integer/rounding math, computed fresh with NumPy
(`exact_match == 1.0`). Rounding half-away-from-zero instead of
half-to-even, forgetting to clamp $x_{\min}/x_{\max}$ to include $0$,
or computing the zero point before dividing by the (correct) scale will
disagree with the oracle on at least one tensor.
