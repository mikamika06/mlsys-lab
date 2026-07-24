## Context

NF4 ("NormalFloat4") is the 4-bit weight format used by QLoRA / `bitsandbytes`.
Instead of a uniform linear grid, its $16$ codes are the quantiles of a
standard normal distribution, so more codes land near zero where weight
values are denser. The fixed, sorted table of $16$ NF4 levels is

$$
L = [
-1.0,\ -0.6961928009986877,\ -0.5250730514526367,\ -0.39491748809814453,\\
-0.28444138169288635,\ -0.18477343022823334,\ -0.09105003625154495,\ 0.0,\\
0.07958029955625534,\ 0.16093020141124725,\ 0.24611230194568634,\ 0.33791524171829224,\\
0.44070982933044434,\ 0.5626170039176941,\ 0.7229568362236023,\ 1.0
].
$$

A quantized tensor of $n$ elements is stored as:

- `idx`: $n$ 4-bit codes (kept here as `uint8` values in $[0, 15]$), one per
  element, indexing into $L$;
- `absmax`: one scale per contiguous block of `block_size` elements (default
  $64$, matching `bitsandbytes`).

An element at flat position $i$, belonging to block $b = \lfloor i /
\text{block\_size} \rfloor$, is reconstructed as

$$
\hat{x}_i = L[\mathrm{idx}_i] \cdot \mathrm{absmax}_b .
$$

This is the **dequantize-only** half of NF4: given the stored codes and
block scales, recover the approximate original tensor. (No quantization —
snapping a float tensor to `idx`/`absmax` — is required here.)

## Task

Implement `nf4_dequantize(idx, absmax, block_size=64)`:

```python
def nf4_dequantize(idx: np.ndarray, absmax: np.ndarray, block_size: int = 64) -> np.ndarray:
    ...
```

- `idx`: 1-D `uint8` array of length $n$ (a multiple of `block_size`), each
  entry in $[0, 15]$.
- `absmax`: 1-D array of length $n / \text{block\_size}$, the per-block
  scale.
- `block_size`: elements per block (default $64$).

Return a 1-D NumPy array of length $n$, `dequant[i] = L[idx[i]] * absmax[i // block_size]`,
using the NF4 table $L$ given above. Do not change the function name,
argument order, or defaults.

## Example

```python
import numpy as np

idx = np.array([0, 7, 15, 7], dtype=np.uint8)
absmax = np.array([2.0], dtype=np.float32)

x = nf4_dequantize(idx, absmax, block_size=4)
# L[0]=-1.0, L[7]=0.0, L[15]=1.0
# x == [-2.0, 0.0, 2.0, 0.0]
```

## What the gate checks

The gate loads a fixed packed tensor (`nf4_idx.npy`, `nf4_absmax.npy`,
`block_size=64`), computes the reference reconstruction directly from the
NF4 table above with NumPy indexing, and compares it elementwise to the
submission's output. The submission passes when the maximum absolute error
is below $10^{-6}$.
