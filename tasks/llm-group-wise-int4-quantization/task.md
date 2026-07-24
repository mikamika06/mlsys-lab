## Context

Quantization reduces the memory footprint of neural network weights by replacing
high precision values with lower precision integer representations. In symmetric
int4 quantization, each quantized value uses 4 bits and represents an integer in
the range $[-8, 7]$.

A group-wise scheme divides the flattened weight tensor into groups of size $g$.
Each group gets its own scale value. For a group $x$ with elements
$x_1, \dots, x_g$, the scale is computed as

$$
s = \frac{\max_i |x_i|}{7}.
$$

Each element is quantized as

$$
q_i = \operatorname{clip}\left(\operatorname{round}\left(\frac{x_i}{s}\right), -8, 7\right).
$$

The dequantized value is reconstructed as

$$
\hat{x}_i = s q_i .
$$

Using smaller groups gives more scale values but usually reduces reconstruction
error because each group can better match the local magnitude distribution.

## Task

Implement `quantize_groupwise_int4(W, group_size)`:

```python
def quantize_groupwise_int4(W: np.ndarray, group_size: int):
    ...
```

The input `W` is a NumPy array of floating point weights with any shape. Flatten
the array in row-major order, split it into consecutive groups of `group_size`
elements, and quantize every group independently.

Return a tuple:

```python
(q, scales, shape)
```

where:

- `q` is a NumPy array of `int8` values containing the flattened int4 codes.
- `scales` is a NumPy array of `float64` group scales.
- `shape` is the original shape tuple so the tensor can be restored.

For groups containing only zeros, use scale $s = 1.0$ and output zero codes.

Do not store packed nibbles. The int4 values should remain as `int8` values in
the output.

## Example

```python
import numpy as np

W = np.array([[-2.0, 0.0, 4.0, 8.0]])
q, scales, shape = quantize_groupwise_int4(W, 2)

# q contains the int4 codes for the two groups
# scales contains one scale per group
# shape is (1, 4)
```

## What the gate checks

The gate computes a NumPy reference implementation of group-wise int4
quantization and compares the returned codes and scales.

The reconstructed tensor is checked using relative error

$$
\mathrm{rel\_err} =
\frac{\lVert \hat{W} - W \rVert_2}
{\lVert W \rVert_2 + 10^{-12}} .
$$

The reconstruction error must stay below the threshold, and the returned
quantization representation must exactly match the reference algorithm.
