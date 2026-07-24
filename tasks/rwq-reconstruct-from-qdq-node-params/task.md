## Context

In the ONNX QDQ (Quantize-Dequantize) representation, a quantized tensor is stored as
three things: integer codes $q$, a scale $s$, and a zero point $z$. A `DequantizeLinear`
node reconstructs the real-valued tensor with

$$
\hat{x} = (q - z) \cdot s .
$$

$s$ and $z$ can be a single scalar shared by the whole tensor (**per-tensor**
quantization), or a 1-D array with one entry per slice along a chosen `axis`
(**per-axis** / per-channel quantization — common for convolution and linear-layer
weights, where each output channel gets its own scale). When per-axis, $s$ and $z$ must
be broadcast so that entry $k$ of $s$ (and $z$) applies to every element of $q$ whose
index along `axis` is $k$.

## Task

Implement `dequantize_linear`:

```python
def dequantize_linear(q: np.ndarray, scale, zero_point, axis: int = 0) -> np.ndarray:
    ...
```

- `q`: integer array of codes, any shape.
- `scale`, `zero_point`: either a Python scalar / 0-D array (per-tensor), or a 1-D
  array-like of length `q.shape[axis]` (per-axis).
- `axis`: the axis `scale`/`zero_point` index into when they are per-axis (ignored when
  they are scalars).

Return $\hat{x} = (q - z) \cdot s$ as a `float64` array with the same shape as `q`,
broadcasting $s$ and $z$ along `axis` when they are per-axis arrays.

## Example

```python
import numpy as np

q = np.array([[0, 128, 255], [10, 20, 30]], dtype=np.uint8)
scale = np.array([0.01, 0.02])       # per-axis, axis=0 -> one scale per row
zero_point = np.array([128, 0])

dequantize_linear(q, scale, zero_point, axis=0)
# row 0: (q - 128) * 0.01 -> [-1.28, 0.0, 1.27]
# row 1: (q - 0)   * 0.02 -> [0.2, 0.4, 0.6]
```

## What the gate checks

The gate builds a NumPy oracle applying the same `(q - z) * s` formula (with the same
broadcasting rule) across several cases: a per-axis-0 fixture (mimicking a stored
per-channel QDQ weight node), a per-tensor scalar case, a per-axis case on a
non-zero axis, and several randomized per-axis cases. `max_abs_err` — the max absolute
error of your reconstruction across every case — must be at most $10^{-6}$.
