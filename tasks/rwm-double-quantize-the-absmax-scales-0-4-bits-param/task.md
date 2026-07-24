## Context

Blockwise 4-bit quantization stores each block of weights using a small codebook and
an absmax scale. For a block $x$ with scale $s$, each value is represented by a
code $q$ from an NF4 codebook and reconstructed as

$$
\hat{x}_i = s \cdot C_{q_i},
$$

where $C$ is the NF4 lookup table.

The block scales themselves consume memory. Double quantization reduces this cost by
quantizing the collection of block scales a second time. If there are $N$ parameters
and block size $b$, single quantization uses approximately

$$
4 + \frac{32}{b}
$$

bits per parameter because every block stores one 32-bit scale. Double quantization
stores 8-bit scale codes and a second-level 32-bit scale:

$$
4 + \frac{8}{b} + \frac{32}{256b}.
$$

The second-level quantization reconstructs the original block scale $s_j$ as

$$
\hat{s}_j = z_j \frac{s_{\max}}{255},
$$

where $z_j$ is the uint8 scale code and $s_{\max}$ is the maximum block scale.

## Task

Implement `double_quant_nf4(W, block_size=64)`.

The function receives a 2-D NumPy array of float values. Split the flattened array into
blocks of `block_size` values. For every block:

1. Compute the absmax scale.
2. Quantize each value to the nearest entry in the NF4 codebook.
3. Quantize all block scales with an 8-bit second-level quantizer.
4. Dequantize the weights using the reconstructed second-level scales.

Return a tuple:

```python
(W_hat, bits_per_param)
```

where `W_hat` is the reconstructed array with the same shape as `W` and
`bits_per_param` is the double-quantized storage estimate.

Use NumPy operations. The returned `bits_per_param` must be a Python `float`.

## Example

```python
import numpy as np

W = np.array([[0.5, -1.0, 2.0, 0.0]], dtype=np.float32)
W_hat, bits = double_quant_nf4(W, block_size=2)
```

The reconstructed values are approximate because both weights and scales are
quantized.

## What the gate checks

The gate builds an independent NumPy oracle for NF4 quantization, second-level scale
quantization, reconstruction, and the bits-per-parameter calculation.

The returned tensor must match the oracle reconstruction within numerical tolerance,
and the reported bits-per-parameter value must match the oracle calculation. A
solution that only performs single-level NF4 quantization fails because its scale
storage and reconstruction differ.
