## Context

The ggml Q4_0 quantization format stores signed 4-bit values in blocks of 32 elements. Each block has its own scale value. For a block $x$ with 32 values, the scale is

$$s = \frac{\max_i |x_i|}{7}.$$

Each quantized value is computed as

$$q_i = \operatorname{clip}\left(\operatorname{round}\left(\frac{x_i}{s}\right), -8, 7\right),$$

and reconstruction uses

$$\hat{x}_i = q_i s.$$

Using one scale for the entire tensor changes the quantization range of every value based on the largest magnitude value in the tensor. When different regions of a tensor have different magnitudes, smaller values lose precision because they share a scale with unrelated large values.

## Task

Implement `q4_0_dequantize(x)`.

```python
def q4_0_dequantize(x: np.ndarray) -> np.ndarray:
    ...
```

The function receives a one-dimensional `float32` or `float64` NumPy array whose length is a multiple of $32$. Return a `float64` NumPy array containing the Q4_0 reconstruction.

The implementation must apply the Q4_0 scale independently for every consecutive block of 32 values. Do not use a single scale computed from the full tensor.

## Example

```python
import numpy as np

x = np.array([0.1, 0.2, 0.3, 0.4] + [0.0] * 28 + [10.0] * 32)
y = q4_0_dequantize(x)

# The first and second groups use different scales.
# A tensor-wide scale would make the small first block inaccurate.
```

## What the gate checks

The gate generates tensors with blocks of different magnitudes and compares the reconstruction error against a NumPy oracle implementing the Q4_0 block algorithm.

The measured metric is the mean squared error

$$\mathrm{MSE} = \frac{1}{n}\sum_i(\hat{x}_i-x_i)^2.$$

A solution using one per-tensor scale produces substantially worse reconstruction on mixed-magnitude tensors and does not pass.
