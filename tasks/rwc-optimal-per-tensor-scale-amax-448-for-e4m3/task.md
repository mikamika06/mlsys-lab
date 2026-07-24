## Context

FP8 inference and training often use a scale factor to map a higher precision tensor
into the representable range of an FP8 format. For the E4M3 format, the largest
finite magnitude is $448$.

Given a tensor $x$, the per-tensor amax is

$$
\mathrm{amax} = \max_i |x_i|.
$$

The standard scaling rule maps the largest absolute value to the largest E4M3
finite value:

$$
s = \frac{\mathrm{amax}}{448}.
$$

The quantization step converts values into FP8 space:

$$
q_i = \mathrm{E4M3}(x_i / s),
$$

and dequantization reconstructs the tensor as

$$
\hat{x}_i = q_i s.
$$

This scaling minimizes clipping because the tensor maximum reaches the available
FP8 range without exceeding it.

## Task

Implement `quantize_fp8_e4m3_amax(x)`:

```python
def quantize_fp8_e4m3_amax(x: np.ndarray) -> tuple[float, np.ndarray]:
    ...
```

The function receives a NumPy array of arbitrary shape and returns:

1. `scale`: a Python float equal to $\mathrm{amax}/448$.
2. `x_hat`: the dequantized FP8 tensor. The algorithm must divide by the scale,
   round each value to E4M3 finite precision, and multiply by the scale again.

Use float64 calculations for the returned reconstruction. Handle an all-zero
tensor by returning scale `1.0` and an all-zero reconstruction.

## Example

```python
import numpy as np

x = np.array([1.0, -100.0, 200.0])
scale, x_hat = quantize_fp8_e4m3_amax(x)

# scale == 200.0 / 448.0
# x_hat contains the FP8 E4M3 quantized and dequantized values
```

## What the gate checks

The gate computes an independent E4M3 quantization oracle. It verifies that the
returned scale matches the oracle's $ \mathrm{amax}/448 $ result and that the
maximum absolute reconstruction error satisfies the required tolerance.

The metric $ \mathrm{max\_abs\_err} $ is computed as

$$
\max_i |\hat{x}_i - \hat{x}^{oracle}_i|.
$$
