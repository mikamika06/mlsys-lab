## Context

NF4 ("4-bit NormalFloat") is the weight format introduced by QLoRA and used by
`bitsandbytes` for 4-bit weight-only quantization. Unlike a uniform integer grid, NF4
uses 16 fixed, non-uniformly spaced levels chosen so that they are information-optimal
for weights that are approximately zero-centered and normally distributed:

$$
L = [-1.0,\ -0.6962,\ -0.5251,\ -0.3949,\ -0.2844,\ -0.1848,\ -0.0911,\ 0.0,\\
\phantom{L = [}0.0796,\ 0.1609,\ 0.2461,\ 0.3379,\ 0.4407,\ 0.5626,\ 0.7230,\ 1.0].
$$

Quantization operates per contiguous **block** of `block_size` elements (block size 64
here, matching the `bitsandbytes` default). For a block $w_b \in \mathbb{R}^{64}$:

1. Compute the block's absolute maximum, $a = \max_i |w_{b,i}|$ (if $a = 0$, use $a = 1$
   to avoid division by zero — every value in the block is already $0$).
2. Normalize: $\tilde{w}_{b,i} = w_{b,i} / a$, so every normalized value lies in
   $[-1, 1]$.
3. Snap each normalized value to the nearest NF4 level, giving a 4-bit index
   $q_{b,i} = \arg\min_k |\tilde{w}_{b,i} - L_k|$.
4. Dequantize: $\hat{w}_{b,i} = L_{q_{b,i}} \cdot a$.

Every block carries its own absmax scale, so the same 16-level codebook adapts to
whatever magnitude range each block happens to have.

## Task

Implement `nf4_quantize_dequantize`:

```python
def nf4_quantize_dequantize(w: np.ndarray, block_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    ...
```

- `w`: 1-D `float64` array whose length is an exact multiple of `block_size`.
- `block_size`: number of elements per quantization block (always 64 in this task).

Return `(indices, dequantized)`:

- `indices`: integer array, same shape as `w`, holding each element's NF4 codebook
  index in `[0, 15]`.
- `dequantized`: `float64` array, same shape as `w`, the reconstructed values
  $\hat{w}_{b,i} = L_{q_{b,i}} \cdot a_b$.

Use the exact NF4 codebook constants above. Handle all-zero blocks (use absmax $=1$
instead of $0$).

## Example

```python
import numpy as np

w = np.concatenate([np.full(64, 0.01), np.full(64, -0.02)])
idx, deq = nf4_quantize_dequantize(w, 64)
# Block 0: absmax=0.01, normalized=1.0 -> nearest level is 1.0 (index 15) -> deq=0.01
# Block 1: absmax=0.02, normalized=-1.0 -> nearest level is -1.0 (index 0) -> deq=-0.02
```

## What the gate checks

The gate builds a NumPy oracle that runs the identical per-block absmax-normalize,
nearest-NF4-level, dequantize pipeline on a fixed test weight vector (covering small
weights, large weights, a uniform-range block, and an all-zero block). It compares:

- `index_exact_match`: your `indices` array must exactly match the oracle's codebook
  indices for every element (must be `1.0`).
- `max_abs_err`: the maximum absolute error between your `dequantized` array and the
  oracle's reconstruction, must be at most $10^{-6}$.
