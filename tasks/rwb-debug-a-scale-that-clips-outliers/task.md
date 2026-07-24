## Context

Symmetric int8 quantization represents floating point values with integer codes and a scale. For a tensor $x$, the quantized values are

$$q_i = \mathrm{clip}\left(\mathrm{round}\left(\frac{x_i}{s}\right), -127, 127\right),$$

and dequantization reconstructs values as

$$\hat{x}_i = q_i s.$$

The scale must cover the largest magnitude value in the tensor. Production quantizers commonly use the absmax rule:

$$s = \frac{\max_i |x_i|}{127}.$$

A scale based on statistics such as $3\sigma$ can work for typical values but fails when rare large values exist. Those values saturate at the int8 limits and cannot be recovered after dequantization.

## Task

Implement `quantize_absmax(x)`:

```python
def quantize_absmax(x: np.ndarray) -> tuple[np.ndarray, float]:
    ...
```

The function takes a NumPy floating point array and returns:

1. An `int8` array of quantized values.
2. A Python float scale.

Use symmetric int8 quantization with the absmax scale:

$$s = \frac{\max(|x|)}{127}.$$

The returned scale must be positive for non-zero inputs. The quantization operation must clip values to the range $[-127, 127]$ before converting to `int8`.

## Example

```python
import numpy as np

x = np.array([-1.0, 0.5, 4.0, -8.0])
q, scale = quantize_absmax(x)

# scale is 8.0 / 127
# q approximately contains [-16, 8, 64, -127]
```

## What the gate checks

The gate computes its own NumPy oracle using the absmax formula and compares the dequantized output:

$$\hat{x} = q \cdot s$$

against the oracle reconstruction.

The relative error

$$\mathrm{rel\_err} = \frac{\lVert \hat{x}_{candidate} - \hat{x}_{oracle} \rVert_2}{\lVert \hat{x}_{oracle} \rVert_2 + 10^{-12}}$$

must be less than $10^{-6}$. Inputs include tensors with large outliers where a scale based on mean or standard deviation will clip values and fail.
