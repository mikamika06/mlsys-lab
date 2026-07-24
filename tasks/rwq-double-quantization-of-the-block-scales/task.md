## Context

NF4 weight quantization stores one fp32 `absmax` scale per block of 64 weights.
That scale array is itself a real memory cost: 32 bits for every 64 quantized
weights is 0.5 bits/param of pure overhead, on top of the 4 bits/param the
weights themselves cost.

QLoRA's *double quantization* treats the absmax array as data to be quantized
again. Because absmax values are strictly non-negative and roughly centered
around their mean, the scheme first subtracts the global mean, then quantizes
the centered array to int8 in blocks of `block_size` (typically 256):

$$
c_1 = \text{absmax} - \mu, \qquad \mu = \text{mean}(\text{absmax})
$$

For each block of `block_size` consecutive values of $c_1$:

$$
s = \frac{\max(|c_1|)}{127}, \qquad
\text{code} = \operatorname{clip}\left(\operatorname{round}\!\left(\frac{c_1}{s}\right), -127, 127\right)
$$

and the absmax array is reconstructed as

$$
\widehat{\text{absmax}} = \text{code} \cdot s + \mu.
$$

Because each second-level block now spends only 8 bits per code plus one
shared fp32 scale, the storage cost per original weight parameter drops from
$32/64 = 0.5$ bits to roughly

$$
\frac{8}{64} + \frac{32}{64 \cdot 256} \approx 0.127 \text{ bits/param},
$$

a saving of about $0.373$ bits/param — the number QLoRA reports.

## Task

Implement `double_quantize_absmax(absmax, block_size=256)`.

`absmax` is a 1-D NumPy array of non-negative fp32 values: one absmax scale
per NF4 weight-quantization block (block size 64, `FIRST_LEVEL_BLOCK_SIZE`).

1. Compute the global mean `mu` of `absmax` and subtract it.
2. Split the centered array into consecutive blocks of `block_size` values
   (the last block may be shorter if `len(absmax)` doesn't divide evenly).
3. For each block, compute `scale = max(|block|) / 127` (use `scale = 1.0`
   when the block is all zero) and quantize to int8 codes in `[-127, 127]`
   via round-to-nearest.
4. Reconstruct: `recon = code * scale + mu`.
5. Compute `bits_saved_per_param`: the fp32 storage cost per original weight
   parameter for the first-level scales alone (`32 / FIRST_LEVEL_BLOCK_SIZE`)
   minus the double-quantized storage cost per original weight parameter
   (`(8*N + 32*n_second_level_blocks + 32) / (N * FIRST_LEVEL_BLOCK_SIZE)`,
   where the extra `+32` accounts for storing `mu` itself and
   `n_second_level_blocks = ceil(N / block_size)`).

Return a 5-tuple:

```python
(codes, scales, mean, recon, bits_saved_per_param)
```

where `codes` is `int8` of shape `(N,)`, `scales` is `float64` of shape
`(ceil(N / block_size),)` (one scale per second-level block), `mean` is a
Python `float`, `recon` is `float64` of shape `(N,)`, and
`bits_saved_per_param` is a Python `float`.

## Example

```python
import numpy as np

absmax = np.abs(np.random.default_rng(0).standard_normal(4096)) * 0.05
codes, scales, mean, recon, bits_saved = double_quantize_absmax(absmax, block_size=256)
# recon has the same shape as absmax and approximates it closely
# bits_saved is close to 0.373 for block_size=256
```

## What the gate checks

The gate rebuilds the same mean-subtract-then-block-quantize scheme with an
independent NumPy oracle and checks two things:

- `rel_err`: the global relative error between your reconstructed `absmax`
  array and the oracle's reconstruction must be at most `1e-5`.
- `bits_abs_err`: your reported `bits_saved_per_param` must be within `0.01`
  of the oracle's value.

A solution that quantizes the raw absmax values directly (skipping the
mean-subtraction step) or that uses a different second-level block size will
diverge from the oracle's reconstruction and fail the `rel_err` gate.
