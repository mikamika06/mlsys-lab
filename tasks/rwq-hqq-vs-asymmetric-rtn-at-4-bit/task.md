## Context

Low-bit weight quantization maps floating point values into a small integer code
range. A common 4-bit asymmetric round-to-nearest (RTN) quantizer uses the
minimum and maximum values to choose scale and zero point:

$$
s = \frac{x_{\max}-x_{\min}}{2^4-1}, \qquad
z = \operatorname{round}\left(-\frac{x_{\min}}{s}\right).
$$

Each value is reconstructed as

$$
\hat{x}=s(\operatorname{clip}(\operatorname{round}(x/s)+z,0,15)-z).
$$

HQQ (Half-Quadratic Quantization) style optimization reduces sensitivity to
outliers by fitting the quantization parameters against an $L_p$ objective with
$0 < p < 1$. For this task, HQQ uses a scalar scale and zero point and selects
the pair minimizing

$$
\sum_i |x_i-\hat{x}_i|^{p}
$$

over a deterministic search grid. The non-convex objective gives more weight to
many small errors instead of letting a few large outliers dominate the parameter
choice.

## Task

Implement `compare_4bit_quantizers(x)`:

```python
def compare_4bit_quantizers(x: np.ndarray) -> tuple[float, float]:
    ...
```

The input is a one-dimensional `float64` NumPy array of weights. Return a pair
`(hqq_mse, rtn_mse)` containing the reconstruction mean squared error of the HQQ
and asymmetric RTN 4-bit quantizers.

For HQQ, search scales in the range from $0.5s_{rtn}$ to $1.5s_{rtn}$ using
101 evenly spaced values. For each scale, compute the best integer zero point
candidate in the range $[-32,32]$ and select the pair with the lowest
$L_{0.7}$ objective. Reconstruct using the selected parameters and return its
MSE.

For asymmetric RTN, use the min/max scale and rounded zero point described in
the context. Use NumPy operations only.

## Example

```python
import numpy as np

x = np.array([-2.0, -1.0, 0.0, 1.0, 12.0])
hqq_mse, rtn_mse = compare_4bit_quantizers(x)

# hqq_mse is lower or equal on this outlier-heavy example
```

## What the gate checks

The gate creates deterministic outlier-heavy weight arrays and computes the
oracle HQQ and asymmetric RTN results with the same mathematical definitions.
The returned HQQ and RTN MSE values must each match the oracle within
$10^{-6}$. A separate gate verifies that the HQQ reconstruction error is not
larger than the RTN reconstruction error on the fixture.
