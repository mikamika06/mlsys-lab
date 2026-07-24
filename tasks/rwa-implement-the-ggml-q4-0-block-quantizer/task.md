## Context

Quantized neural network formats reduce memory usage by storing low-bit representations of weights. The ggml Q4_0 format stores values in blocks of 32 elements. Each block has one scale value stored as fp16 and 32 signed 4-bit quantized values.

For a block $x \in \mathbb{R}^{32}$, Q4_0 computes

$$
d = \frac{\max_i |x_i|}{-8}.
$$

The quantized value is computed by rounding and clamping:

$$
q_i = \operatorname{clip}\left(\operatorname{round}\left(\frac{x_i}{d}\right), -8, 7\right).
$$

The dequantized approximation is

$$
\hat{x}_i = d q_i.
$$

The 32 four-bit values are stored as unsigned nibbles. A signed value $q_i$ in the range $[-8, 7]$ is represented by adding $8$, producing a nibble in the range $[0, 15]$. Two nibbles are packed into each byte.

## Task

Implement `q4_0_quantize(x)`:

```python
def q4_0_quantize(x: np.ndarray):
    ...
```

The input is a one-dimensional NumPy array whose length is a multiple of $32$.

Return a tuple `(scales, codes)`:

- `scales` is a NumPy array of shape `(n_blocks,)` with dtype `np.float16`.
- `codes` is a NumPy array of shape `(n_blocks, 16)` with dtype `np.uint8`.
- Each row of `codes` stores one 32-element block. Byte $j$ contains two packed nibbles:
  the low nibble stores element $2j$ and the high nibble stores element $2j+1$.
- The quantization must follow the Q4_0 equations above.

Do not change the function name or return format.

## Example

```python
import numpy as np

x = np.array([
    -1.0, 0.0, 1.0, 2.0,
] * 8, dtype=np.float32)

scales, codes = q4_0_quantize(x)

# scales contains one fp16 scale for the 32 values.
# codes contains 16 bytes of packed four-bit values.
```

## What the gate checks

The gate builds a NumPy oracle implementation of the Q4_0 algorithm, dequantizes both the oracle result and the submitted result, and computes the mean squared error

$$
\mathrm{MSE} = \frac{1}{n}\sum_i(\hat{x}_i^{student}-\hat{x}_i^{oracle})^2 .
$$

The submitted implementation passes when the MSE is at most $10^{-8}$. The check uses independent generated numeric inputs so the implementation must follow the quantization procedure rather than matching fixed examples.
