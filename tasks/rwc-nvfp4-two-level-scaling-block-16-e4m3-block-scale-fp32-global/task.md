## Context

NVFP4 is a low-precision floating point representation that uses two levels of scaling. Values are stored as 4-bit $e2m1$ elements. Groups of 16 elements share a block scale stored in $e4m3$, and all block scales share one tensor-level $fp32$ scale.

For an element $x_i$ in a block, reconstruction follows

$$
\hat{x}_i = q_i \cdot s_b \cdot g,
$$

where $q_i$ is the quantized $e2m1$ value, $s_b$ is the decoded $e4m3$ block scale, and $g$ is the global $fp32$ scale.

The $e2m1$ value set used here is

$$
\{0, \pm0.5, \pm1, \pm1.5, \pm2, \pm3, \pm4, \pm6\}.
$$

The block size is fixed at 16 values. The global scale normalizes the largest block scale so that the stored block scales fit in the $e4m3$ range.

## Task

Implement `quantize_nvfp4(x)`:

```python
def quantize_nvfp4(x: np.ndarray):
    ...
```

The input is a one-dimensional NumPy array of `float32` values. Return a tuple:

```python
(codes, block_scales, global_scale, reconstruction)
```

where:

- `codes` is a `uint8` array containing the 4-bit $e2m1$ code for each input element.
- `block_scales` is a `uint8` array containing the encoded $e4m3$ scale for each block of 16 values.
- `global_scale` is a Python float containing the tensor-level $fp32$ scale.
- `reconstruction` is a `float32` array containing the reconstructed values.

Use the following algorithm:

1. Compute the maximum absolute input value.
2. Choose `global_scale` so that the largest block scale can be represented by $e4m3$.
3. Split the input into blocks of 16 values.
4. Compute one block scale per block, quantize it to $e4m3`, and use the decoded value during reconstruction.
5. Quantize each value divided by its block scale and global scale to the nearest $e2m1` value.
6. Return the encoded values and the reconstructed tensor.

## Example

```python
import numpy as np

x = np.array([1.0, -2.0, 0.5, 0.0], dtype=np.float32)
codes, scales, g, y = quantize_nvfp4(x)

# codes and scales are compact floating point representations.
# y is the decoded approximation of x.
```

## What the gate checks

The gate computes the same NVFP4 algorithm with a NumPy oracle and checks three properties.

`code_exact` requires the returned $e2m1$ codes to exactly match the oracle.

`scale_max_abs_err` measures the largest absolute difference between the returned decoded block scales and the oracle decoded block scales.

`max_abs_err` measures the largest absolute difference between the returned reconstruction and the oracle reconstruction.
