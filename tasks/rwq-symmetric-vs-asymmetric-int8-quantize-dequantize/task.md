## Context

Quantization maps a real‐valued tensor $x \in \mathbb{R}^n$ to an integer representation that can be stored in fewer bits.  
For 8‑bit signed integers we typically use the *symmetric* scheme:

$$\text{scale}_{\text{s}} = \frac{\max_i |x_i|}{127}, \qquad
q_{\text{s}} = \operatorname{round}\!\left(\frac{x}{\text{scale}_{\text{s}}}\right), \qquad
q_{\text{s}}\in[-128,127]$$

The dequantized value is then $x_{\text{dq}} = q_{\text{s}}\;\text{scale}_{\text{s}}$.

In the *asymmetric* (zero‑point) scheme we allow a non‑zero offset:

$$
\begin{aligned}
\text{min} &= \min_i x_i, & \text{max} &= \max_i x_i,\\
\text{scale}_{\text{a}} &= \frac{\text{max}-\text{min}}{255}, &
z_p &= \operatorname{round}\!\left(-\,\frac{\text{min}}{\text{scale}_{\text{a}}}\right),\\[4pt]
q_{\text{a}} &= \operatorname{round}\!\left(\frac{x}{\text{scale}_{\text{a}}}+z_p\right),
& q_{\text{a}}\in[0,255],\\
x_{\text{dq}} &= (q_{\text{a}}-z_p)\;\text{scale}_{\text{a}}.
\end{aligned}
$$

Both schemes are used in machine‑learning inference to reduce memory and bandwidth.  
The task is to implement the quantize–dequantize roundtrip for each scheme.

## Task

Implement two functions:

```python
def sym_quant_dequant(x: np.ndarray) -> np.ndarray:
    """Return a float64 array that is the dequantized result of symmetric INT8."""
```

```python
def asym_quant_dequant(x: np.ndarray) -> np.ndarray:
    """Return a float64 array that is the dequantized result of asymmetric INT8."""
```

Both functions must:

1. Accept any 2‑D or higher NumPy array `x` with real values.
2. Use only NumPy operations (no explicit Python loops).
3. Return a new array of type `float64` having the same shape as `x`.
4. Quantize to signed 8‑bit (`int8`) for symmetric and unsigned 8‑bit (`uint8`) for asymmetric, then dequantize back to float.

The grader will compare your outputs against a NumPy oracle and enforce numerical accuracy.

## Example

```python
import numpy as np

x = np.array([[0.0, -1.5], [2.3, 4.7]])
sym_dq = sym_quant_dequant(x)
asym_dq = asym_quant_dequant(x)

print(sym_dq)
# [[ 0.          -1.4925373134328356]
#  [ 2.291666666666667   4.708333333333334]]
print(asym_dq)
# [[ 0.          -1.5       ]
#  [ 2.3         4.7        ]]
```

The asymmetric result is exact because the input values lie on a uniform grid that matches the chosen scale and zero‑point.

## What the gate checks

Three metrics are evaluated:

| Metric | Description | Threshold |
|--------|-------------|-----------|
| `max_abs_err` | The maximum absolute difference between your dequantized arrays and the oracle’s, over all test cases. | ≤ $10^{-6}$ |
| `sym_mse_diff` | The largest absolute difference between the mean‑squared error of your symmetric result and that of the oracle. | ≤ $10^{-9}$ |
| `asym_mse_diff` | Same as above for the asymmetric scheme. | ≤ $10^{-9}$ |

All three must pass for the solution to be accepted.
