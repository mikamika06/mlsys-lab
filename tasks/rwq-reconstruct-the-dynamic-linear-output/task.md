## Context

Dynamic quantization for a linear layer stores weights as `int8` values with
per-channel (per-output-row) scales $s_w \in \mathbb{R}^{m}$, and quantizes
activations on-the-fly using a per-tensor scale $s_x$ and zero-point $z_x$.

Given a quantized input vector $\hat{x} \in \mathbb{Z}^{n}$ (stored as `uint8`)
and quantized weight matrix $\hat{W} \in \mathbb{Z}^{m \times n}$ (stored as
`int8`), the reconstructed float output is:

$$y = \bigl(\hat{W} \cdot (\hat{x} - z_x)\bigr) \cdot s_w \cdot s_x$$

where:
- $\hat{x} - z_x$ dequantizes each activation to a signed integer,
- $\hat{W} \cdot (\hat{x} - z_x)$ is the integer matrix-vector product (done in
  float64 for correctness here),
- $s_w$ is the per-output-channel weight scale (shape $(m,)$), broadcast
  element-wise along the output dimension,
- $s_x$ is the scalar activation scale.

The result $y \in \mathbb{R}^{m}$ is the approximate float output of the linear
layer.

## Task

Implement `dequant_linear_output(W_int8, w_scales, x_uint8, x_scale, x_zp)`:

```python
def dequant_linear_output(W_int8, w_scales, x_uint8, x_scale, x_zp):
    ...
```

- `W_int8`: `int8` NumPy array of shape `(m, n)` — quantized weight matrix.
- `w_scales`: `float32` NumPy array of shape `(m,)` — per-output-channel weight scales.
- `x_uint8`: `uint8` NumPy array of shape `(n,)` — quantized input activations.
- `x_scale`: scalar float — per-tensor activation scale.
- `x_zp`: scalar int — activation zero-point.

Return a `float32` NumPy array of shape `(m,)` with the reconstructed output:
$$y_i = \left(\sum_{j} \hat{W}_{ij} \cdot (\hat{x}_j - z_x)\right) \cdot s_{w,i} \cdot s_x$$

## Example

```python
import numpy as np
W = np.array([[1, -1], [2, 0]], dtype=np.int8)
w_s = np.array([0.5, 0.25], dtype=np.float32)
x = np.array([130, 126], dtype=np.uint8)   # zp=128 -> signed [-2, 2] -> wait: 130-128=2, 126-128=-2
x_scale = 0.1
x_zp = 128
out = dequant_linear_output(W, w_s, x, x_scale, x_zp)
# row0: (1*2 + (-1)*(-2))*0.5*0.1 = 4*0.05 = 0.2
# row1: (2*2 + 0*(-2))*0.25*0.1  = 4*0.025 = 0.1
# out ≈ [0.2, 0.1]
```

## What the gate checks

The grader generates random int8 weight matrices, float weight scales, uint8
activations with a known zero-point, and a scalar activation scale. It computes
the reference output using the same formula above in float64, then checks that
your implementation's relative L2 error is $\le 10^{-4}$.
