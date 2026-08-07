## Context

Key-value caches in transformer inference often use reduced precision storage to lower memory usage. A quantizer maps floating point values to integer levels and reconstructs them during attention computation.

For a key matrix $K \in \mathbb{R}^{T \times C}$, symmetric quantization with $b$ bits uses

$$
q = \operatorname{round}\left(\frac{x}{s}\right), \qquad \hat{x} = q s ,
$$

where $s$ is the quantization scale.

A per-token implementation computes one scale for each token:

$$
s_i = \frac{\max_j |K_{ij}|}{2^{b-1}-1}.
$$

This allows large-magnitude channels to dominate the scale of the whole token. Smaller channels then lose precision.

The production approach for key caches uses per-channel scales:

$$
s_j = \frac{\max_i |K_{ij}|}{2^{b-1}-1}.
$$

Each channel gets its own range, preserving precision when some channels contain outliers.

## Task

Implement `quantize_keys_per_channel(K, bits=4)`.

The function receives a list of lists of floats `K` with shape $(T, C)$ and returns a list of the same shape containing the dequantized keys.

Requirements:

- Use symmetric uniform quantization.
- Compute one scale per channel, not one scale per token.
- Use the formula $s_j = \max_i |K_{ij}|/(2^{b-1}-1)$.
- Return a `float64` array.

## Example

```python

K = [
    [10.0, 0.2, 0.1],
    [12.0, 0.1, 0.3],
    [11.0, 0.2, 0.2],
]

out = quantize_keys_per_channel(K, bits=4)
```

The first channel has a much larger magnitude than the remaining channels. Per-channel quantization avoids forcing all channels to use the first channel's scale.

## What the gate checks

The gate builds a Python reference implementation of per-channel quantization and compares the submitted output against it.

The reported $mse$ is

$$
\operatorname{MSE} = \frac{1}{TC}\sum_{i,j}(\hat{K}_{ij}^{\mathrm{submitted}}-\hat{K}_{ij}^{\mathrm{reference}})^2 .
$$

The reference solution must produce zero error. A per-token implementation fails because its reconstruction uses different scales and does not match the oracle.
