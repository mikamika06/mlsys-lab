## Context

KV cache quantization reduces the memory cost of transformer inference by storing keys and
values in lower precision. MLX-style KV quantization uses group-wise asymmetric affine
quantization. Each group of values has its own scale and zero-point.

For a group of values $x$, with $b$ quantization bits, the integer range is

$$
0 \leq q \leq 2^b - 1 .
$$

The affine quantizer computes

$$
s = \frac{x_{\max} - x_{\min}}{2^b - 1},
$$

and the zero-point is chosen so that the minimum value maps close to zero:

$$
z = \mathrm{round}\left(-\frac{x_{\min}}{s}\right).
$$

Each value is quantized as

$$
q = \mathrm{clip}(\mathrm{round}(x / s + z), 0, 2^b - 1),
$$

and reconstructed as

$$
\hat{x} = s(q-z).
$$

The KV cache tensor is split into consecutive groups along the last dimension. A separate
scale and zero-point are stored for every group.

## Task

Implement `quantize_kv_group_affine(kv, kv_bits, kv_group_size)`.

The function receives a NumPy array `kv` containing KV cache values and returns a tuple:

```python
q, scales, zeros = quantize_kv_group_affine(kv, kv_bits, kv_group_size)
```

where:

- `q` contains the quantized integer values.
- `scales` contains one scale per group.
- `zeros` contains one zero-point per group.

Groups are formed by splitting the last dimension into chunks of size `kv_group_size`.
The output must represent the affine quantizer described above. The function must support
inputs whose last dimension is divisible by `kv_group_size`.

The returned quantized values must be integer arrays. The scales and zero-points must
contain enough information for dequantization.

## Example

```python
import numpy as np

kv = np.array([[[-1.0, 1.0, 3.0, 5.0]]])

q, scales, zeros = quantize_kv_group_affine(kv, 4, 2)

x_hat = scales * (q - zeros)
```

The two groups are `[-1, 1]` and `[3, 5]`. Each group has its own affine parameters.

## What the gate checks

The grader computes a NumPy reference implementation of the group affine quantizer and
compares the student's dequantized output against the oracle.

The `mse` score is the mean squared error between the student's dequantized KV cache and
the reference dequantized KV cache. It must satisfy

$$
\mathrm{MSE} \leq 10^{-8}.
$$

The grader also computes a small attention-style matrix product using the quantized KV
values. The resulting `attention_mse` must satisfy

$$
\mathrm{MSE}_{attention} \leq 10^{-8}.
$$
