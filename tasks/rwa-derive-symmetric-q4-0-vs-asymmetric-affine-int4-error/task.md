## Context

Low-bit weight and activation quantization maps floating point values to a small
integer range and reconstructs them using a scale and sometimes a zero point.

A symmetric Q4_0 quantizer uses a signed 4-bit range with values
$q \in [-8, 7]$. For a block with maximum absolute value
$ a_{\max} = \max_i |x_i| $, its step size is

$$
s = \frac{2a_{\max}}{15}.
$$

Each element is quantized by rounding

$$
q_i = \mathrm{clip}\left(\mathrm{round}\left(\frac{x_i}{s}\right), -8, 7\right),
$$

and reconstructed as

$$
\hat{x}_i = s q_i .
$$

An asymmetric affine INT4 quantizer uses the unsigned range
$q \in [0, 15]$. For minimum value $x_{\min}$ and maximum value $x_{\max}$,

$$
s = \frac{x_{\max}-x_{\min}}{15},
$$

and the zero point is

$$
z = \mathrm{round}\left(-\frac{x_{\min}}{s}\right).
$$

The quantized values and reconstruction are

$$
q_i = \mathrm{clip}\left(\mathrm{round}\left(\frac{x_i}{s}+z\right),0,15\right),
$$

$$
\hat{x}_i = s(q_i-z).
$$

For skewed data, the asymmetric range can use the available integer levels more
effectively because it does not waste half of the range representing unused
negative values.

## Task

Implement `compare_q4_errors(block)`:

```python
def compare_q4_errors(block: np.ndarray) -> dict:
    ...
```

The function receives a one-dimensional floating point NumPy array representing
a quantization block. Return a dictionary with exactly these keys:

- `"q4_0_error"`: the relative reconstruction error of the symmetric Q4_0
  quantizer.
- `"affine_int4_error"`: the relative reconstruction error of the asymmetric
  affine INT4 quantizer.
- `"winner"`: the string `"q4_0"` or `"affine_int4"` corresponding to the lower
  error.

The relative reconstruction error is

$$
\frac{\lVert \hat{x}-x\rVert_2}{\lVert x\rVert_2 + 10^{-12}} .
$$

Use NumPy operations for the quantization calculations.

## Example

```python
import numpy as np

x = np.array([-1.0, 0.2, 0.3, 0.5, 2.0, 4.0])
result = compare_q4_errors(x)

# result contains:
# {
#   "q4_0_error": ...,
#   "affine_int4_error": ...,
#   "winner": "affine_int4"
# }
```

## What the gate checks

The gate computes the same two quantizers with an independent NumPy oracle on
several skewed quantization blocks. It compares the two reported error values
and the selected winner against the oracle result. The combined numerical
relative error metric `$rel\_err$` must satisfy

$$
rel\_err \le 10^{-4}.
$$

A solution that uses the wrong scale formula, misses the affine zero point, or
returns the wrong winning scheme will fail.
