## Context

Affine (min–max) quantization maps a floating-point range $[x_{\min}, x_{\max}]$ to $b$-bit integers $\{0, 1, \ldots, 2^b - 1\}$.
The **scale** converts between the two domains:

$$s = \frac{x_{\max} - x_{\min}}{2^b - 1}$$

Given scale $s$ and zero-point $z = \text{round}(-x_{\min} / s)$, the quantize–dequantize round-trip is:

$$\hat{x} = s \cdot \bigl(\text{clip}(\text{round}(x / s - z),\, 0,\, 2^b-1) + z\bigr)$$

A common bug replaces the denominator $2^b - 1$ with $2^b$, i.e.

$$s_{\text{buggy}} = \frac{x_{\max} - x_{\min}}{2^b}$$

This causes the top quantization level to be unreachable and inflates the reconstruction error — the maximum value $x_{\max}$ can never be exactly represented.

## Task

Implement `affine_quant_dequant(x, bits)`:

```python
def affine_quant_dequant(x: np.ndarray, bits: int) -> np.ndarray:
    ...
```

- `x`: a 1-D float64 NumPy array.
- `bits`: number of quantization bits (e.g., 8).
- Returns a float64 array of the same shape: the dequantized reconstruction using the **correct** denominator $2^b - 1$.

The zero-point is $z = 0$ (symmetric-ish: just shift by min). Use:
$$s = \frac{\max(x) - \min(x)}{2^b - 1}, \quad q = \text{clip}\!\left(\text{round}\!\left(\frac{x - \min(x)}{s}\right), 0, 2^b - 1\right), \quad \hat{x} = q \cdot s + \min(x)$$

## Example

```python
import numpy as np
x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
x_hat = affine_quant_dequant(x, bits=2)
# With 2 bits: levels 0,1,2,3 -> scale = 3/3 = 1.0
# x_hat: [0., 1., 2., 3.]  (perfect reconstruction)
```

A buggy version using $s = 3/4 = 0.75$ would give $\hat{x} = [0., 0.75, 1.5, 2.25]$.

## What the gate checks

The grader computes the reference dequantized output with the correct $2^b - 1$ denominator and compares it against the student's output. The **max_abs_err** must be $\le 10^{-6}$.
