## Context

Quantization stores weights using lower precision values and auxiliary metadata. A common scheme represents each weight as a signed 4-bit integer and stores a scale and zero-point for each group of weights.

A packed byte can store two int4 values. For a byte $p$, the lower and upper nibbles are

$$
q_0 = p \mathbin{\&} 15, \qquad q_1 = (p \gg 4) \mathbin{\&} 15 .
$$

Each unsigned nibble is interpreted as a quantized value $q \in \{0,\dots,15\}$. The reconstructed value for a weight in group $g$ is

$$
w = (q - z_g) s_g,
$$

where $s_g$ is the scale and $z_g$ is the zero-point for that group.

The group assignment is determined by the original flattened weight index. If the group size is $G$, the index $i$ uses group

$$
g = \left\lfloor \frac{i}{G} \right\rfloor .
$$

This task models a common inference path: unpack compact int4 storage, then apply group-wise dequantization.

## Task

Implement `dequantize_int4`:

```python
def dequantize_int4(packed, scales, zero_points, shape, group_size):
    ...
```

The arguments are:

- `packed`: a 1-D NumPy array of `uint8` values. Each byte stores two consecutive quantized weights, with the low nibble first.
- `scales`: a 1-D NumPy array of `float64` scales, one value per group.
- `zero_points`: a 1-D NumPy array of `int64` zero-points, one value per group.
- `shape`: the output tensor shape.
- `group_size`: the number of weights covered by one scale and zero-point pair.

Return a NumPy `float64` array with the requested shape containing the reconstructed weights. Use the unpacking rule and dequantization equation from the context.

## Example

```python
import numpy as np

packed = np.array([0x21, 0xF3], dtype=np.uint8)
scales = np.array([0.5], dtype=np.float64)
zero_points = np.array([0], dtype=np.int64)

out = dequantize_int4(packed, scales, zero_points, (4,), 4)
# [0.5, 1.0, 1.5, 7.5]
```

## What the gate checks

The gate creates several packed int4 tensors, computes the expected reconstruction by independently unpacking the nibbles and applying the group scales and zero-points, and compares the submitted function against that NumPy reference.

The reported metric is $\mathrm{max\_abs\_err}$, defined as

$$
\max_i |w_i - \hat{w}_i|.
$$

The solution passes when $\mathrm{max\_abs\_err} < 10^{-6}$.
