## Context

KV-cache quantization has to pick a *grouping axis*: which set of elements
shares one scale/zero-point. For a key cache $K \in \mathbb{R}^{n \times d}$
(one row per token, one column per RoPE-rotated channel), there are two
natural choices:

- **per-channel**: one affine quantizer per column, calibrated over all $n$
  tokens.
- **per-token**: one affine quantizer per row, calibrated over all $d$
  channels.

For a uniform affine min-max quantizer with $b$ bits, a group with values
$x$ is quantized as

$$
s = \frac{\max(x) - \min(x)}{2^b - 1}, \qquad
z = \operatorname{round}\!\left(\frac{-\min(x)}{s}\right)
$$

$$
\hat{x} = \left(\operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{x}{s} + z\right),\, 0,\, 2^b-1\right) - z\right) \cdot s
$$

KIVI's key empirical finding is that real RoPE-rotated keys have **channels
with a persistent, channel-specific bias** (some channels sit at a
consistently different magnitude than others, across *every* token) with
only small residual variation *within* a channel. Grouping by channel
lets each channel's quantizer calibrate to its own tight residual spread.
Grouping by token forces one quantizer to span the *entire* range of
channel biases in that row, so the token-wide step size $s$ becomes coarse
and crushes every channel in that row down to a handful of shared
buckets — including channels whose own residual spread was tiny.

## Task

Implement `per_channel_vs_per_token_k_mse(K, bits)`:

```python
def per_channel_vs_per_token_k_mse(K: np.ndarray, bits: int) -> np.ndarray:
    ...
```

- `K`: `(n_tokens, d_channels)` float64 key cache.
- `bits`: quantizer bit-width for the uniform affine min-max quantizer
  above.

Return `np.array([mse_per_channel, mse_per_token])`:

- `mse_per_channel`: mean squared reconstruction error of `K` quantized
  **per-channel** (one scale/zero-point per column, min/max over all rows).
- `mse_per_token`: mean squared reconstruction error of `K` quantized
  **per-token** (one scale/zero-point per row, min/max over all columns).

## Example

```python
K.shape  # (32, 16), channels have distinct biases, small per-token noise
per_channel_vs_per_token_k_mse(K, bits=4)
# -> array([0.000125, 0.072768])
# per-channel MSE is orders of magnitude below per-token MSE: each
# channel's quantizer only has to cover its own small residual spread,
# while the per-token quantizer must cover the full spread of channel
# biases in every row.
```

## What the gate checks

The gate builds several `K` matrices from a seeded generator: each has a
per-channel bias drawn from a wide range plus small Gaussian noise, at
varying `(n_tokens, d_channels, bits)`. For each case it computes the real
reference `mse_per_channel` / `mse_per_token` with a NumPy oracle
implementing the exact quantize/dequantize formula above along each axis.

Your two returned numbers must (a) be within relative error `1e-6` of the
oracle's, **and** (b) satisfy `mse_per_channel < mse_per_token` — the
actual KIVI finding the task is about. A solution that gets the formula
right but groups along the wrong axis (e.g. swaps per-channel and
per-token, or uses a single per-tensor scale for both) will either miss
the numeric tolerance or violate the ordering, and fails either way.
