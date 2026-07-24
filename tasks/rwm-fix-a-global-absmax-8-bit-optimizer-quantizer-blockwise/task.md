## Context

Low precision optimizer states reduce memory usage by storing values with fewer bits. A simple 8-bit quantizer stores integer codes and a scale derived from the largest magnitude value.

For a tensor $x$, global absmax scaling uses

$$
s = \frac{\max_i |x_i|}{127},
$$

and quantizes each element as

$$
q_i = \operatorname{clip}\left(\operatorname{round}\left(\frac{x_i}{s}\right), -127, 127\right).
$$

The reconstructed tensor is

$$
\hat{x}_i = q_i s.
$$

A problem occurs when optimizer states contain localized spikes. A single large value controls the scale for every element, causing small values to lose precision. Production optimizers avoid this by dividing the state into blocks and storing an independent scale for each block.

For a block $b$ with values $x_b$, blockwise scaling uses

$$
s_b = \frac{\max_{i \in b}|x_i|}{127},
$$

then applies the same quantization rule only inside that block.

## Task

Implement `blockwise_quantize_dequantize(x, block_size)`.

The function must:

1. Accept a one-dimensional NumPy array `x`.
2. Split the array into consecutive blocks of at most `block_size` elements.
3. Compute one absmax scale per block.
4. Quantize each block to signed 8-bit integer values.
5. Dequantize the blocks back to a `float64` NumPy array.
6. Return only the reconstructed array.

Use NumPy operations for the numeric work. If a block has maximum magnitude zero, it must reconstruct as all zeros.

The function signature is:

```python
def blockwise_quantize_dequantize(x: np.ndarray, block_size: int) -> np.ndarray:
    ...
```

## Example

```python
import numpy as np

x = np.array([0.1, 0.2, 10.0, 0.3, 0.4])
y = blockwise_quantize_dequantize(x, 2)

# The first two values use their own scale, the spike 10.0
# only affects its block.
```

## What the gate checks

The gate creates optimizer-like state arrays containing normal values and a few localized spikes.

It computes the oracle result by independently applying blockwise absmax quantization and dequantization with NumPy. The submitted implementation must produce a reconstruction whose relative error

$$
\mathrm{rel\_err} =
\frac{\lVert \hat{x}_{candidate} - \hat{x}_{oracle}\rVert_2}
{\lVert \hat{x}_{oracle}\rVert_2 + 10^{-12}}
$$

is at most $10^{-6}$.

A global-absmax implementation fails because spikes incorrectly determine the scale for unrelated blocks.
