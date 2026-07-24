## Context

Weight-only quantization replaces floating point weights with a smaller integer representation. A simple symmetric $b$-bit quantizer maps values using a scale factor:

$$
s = \frac{\max(|W|)}{2^{b-1}-1}, \qquad \hat{W} = \operatorname{clip}\left(\operatorname{round}\left(\frac{W}{s}\right), -2^{b-1}, 2^{b-1}-1\right)s .
$$

A single scale for the whole matrix can be dominated by a few large values. Activation-aware weight quantization (AWQ) reduces this effect by using separate scales for important channels. For a weight matrix $W \in \mathbb{R}^{m \times d}$, input channels correspond to columns of $W$. Per-channel scaling gives each column its own quantization range:

$$
s_j = \frac{\max_i |W_{ij}|}{2^{b-1}-1}, \qquad
\hat{W}_{ij} = \operatorname{round}\left(\frac{W_{ij}}{s_j}\right)s_j .
$$

The quantized matmul result should approximate the floating point reference:

$$
Y = WX, \qquad \hat{Y} = \hat{W}X .
$$

Using per-channel scales protects salient channels because large columns no longer consume the quantization range of unrelated channels.

## Task

Implement `awq_matmul(W, X)`:

```python
def awq_matmul(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    ...
```

The function receives a weight matrix `W` with shape $(m, d)$ and an input matrix `X` with shape $(d, n)$. Simulate a 4-bit AWQ weight round-trip:

1. Compute one symmetric quantization scale per input channel of `W`.
2. Quantize each column to signed 4-bit integer values.
3. Dequantize the weights back to floating point.
4. Return the dequantized matrix multiplication result.

Use NumPy operations. The returned array must contain `float64` values.

## Example

```python
import numpy as np

W = np.array([[10.0, 0.5], [8.0, -0.4]])
X = np.array([[1.0], [2.0]])

Y_hat = awq_matmul(W, X)
```

The result should be close to the full precision value `W @ X`, while using per-channel quantization instead of a single matrix-wide scale.

## What the gate checks

The gate builds several matrices with salient high-magnitude channels, computes the full precision oracle result $WX$, and measures

$$
\mathrm{rel\_err} =
\frac{\lVert \hat{Y} - Y \rVert_2}{\lVert Y \rVert_2 + 10^{-12}} .
$$

The returned result must satisfy $\mathrm{rel\_err} \le 0.15$. A plain matrix-wide 4-bit quantizer loses accuracy on the same cases and does not pass.
