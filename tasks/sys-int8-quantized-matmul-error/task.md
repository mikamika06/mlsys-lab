## Context

Inference systems often reduce memory bandwidth by storing activations and weights in lower precision formats. In symmetric INT8 quantization, a floating point tensor is represented by integer values and a scale factor.

For a tensor $X$, the quantization rule is

$$
q_X = \operatorname{clip}(\operatorname{round}(X / s_X), -127, 127),
$$

where the scale is chosen as

$$
s_X = \frac{\max(|X|)}{127}.
$$

The original value is approximated by $X \approx s_X q_X$. For matrix multiplication, two INT8 matrices can be multiplied using INT32 accumulation:

$$
C_{\mathrm{int32}} = q_A q_B,
$$

then converted back to floating point with

$$
\hat{C} = C_{\mathrm{int32}}(s_A s_B).
$$

Using INT32 accumulation is important because the product sum can exceed the range of INT8 values during the dot product.

## Task

Implement `quantized_matmul(A, B)`:

```python
def quantized_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    ...
```

The inputs are 2-D NumPy arrays with compatible matrix multiplication shapes. Quantize each input independently using symmetric INT8 quantization with scales computed from the maximum absolute value of each input tensor. Perform the matrix multiplication using INT32 accumulation and return the dequantized floating point result.

The returned array should contain the approximate result of `A @ B`. Use NumPy operations only.

## Example

```python
import numpy as np

A = np.array([[1.0, -2.0], [0.5, 3.0]])
B = np.array([[2.0, 1.0], [-1.0, 4.0]])

C = quantized_matmul(A, B)

# C is close to:
# [[ 4.0, -7.0],
#  [-2.0, 12.5]]
```

## What the gate checks

The gate computes a NumPy reference implementation of INT8 quantized matrix multiplication using the same quantization definition and compares the submitted result with a relative error metric.

The score is

$$
\mathrm{rel\_err} =
\frac{\lVert \hat{C} - C_{\mathrm{ref}}\rVert_2}
{\lVert C_{\mathrm{ref}}\rVert_2 + 10^{-12}} .
$$

The implementation passes when $\mathrm{rel\_err} \le 0.02$.
