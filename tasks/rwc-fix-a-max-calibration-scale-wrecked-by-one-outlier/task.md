## Context

Quantized inference often calibrates an activation tensor by finding an absolute maximum value
($\mathrm{amax}$) and mapping the range to integer codes. A symmetric int8 calibration with
maximum code $q_{\max}$ uses

$$
\mathrm{scale} = \frac{\mathrm{amax}}{q_{\max}},
$$

and reconstruction uses

$$
\hat{x} = \mathrm{clip}\left(\mathrm{round}\left(\frac{x}{\mathrm{scale}}\right),
-q_{\max}, q_{\max}\right)\mathrm{scale}.
$$

A max-based calibration is sensitive to a single abnormal activation. If one value is much
larger than the rest of the tensor, then

$$
\mathrm{amax} = \max_i |x_i|
$$

can make the scale too large. Normal values are then represented by only a few integer levels,
increasing reconstruction error.

A common calibration fix is percentile calibration. Instead of the largest value, choose a high
percentile of absolute activations:

$$
\mathrm{amax} = \mathrm{percentile}(|x|, p).
$$

This ignores a small number of extreme values while preserving the useful dynamic range of the
typical activations.

## Task

Implement `calibrate_scale_and_dequantize(x, qmax=127, percentile=99.0)`.

The function receives a one-dimensional NumPy array of floating point activations and returns a
tuple:

```python
(
    amax,
    scale,
    reconstructed
)
```

where:

- `amax` is the percentile-based calibration value computed from `abs(x)`.
- `scale` is `amax / qmax`.
- `reconstructed` is the dequantized tensor using symmetric integer quantization with clipping.

Use NumPy operations. The returned reconstruction should use the same dtype-independent numerical
behavior as a float64 computation.

## Example

```python
import numpy as np

x = np.array([0.1, -0.2, 0.3, 100.0])

amax, scale, reconstructed = calibrate_scale_and_dequantize(
    x, qmax=127, percentile=75.0
)

# amax is based on np.percentile(np.abs(x), 75.0)
# reconstruction uses the resulting scale
```

## What the gate checks

The gate builds an outlier-contaminated calibration tensor and computes the oracle result using
NumPy percentile calibration. It checks that the returned `amax` matches the oracle value with
maximum absolute error

$$
\max_i |a_i - b_i| \le 10^{-9}.
$$

It also checks the reconstruction relative error against the percentile-calibrated oracle:

$$
\frac{\lVert \hat{x}-x\rVert_2}{\lVert x\rVert_2} \le 0.15.
$$

A solution that still uses the maximum activation for calibration fails because the lone outlier
causes a much worse scale.
