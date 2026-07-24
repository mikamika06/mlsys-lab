## Context

8-bit optimizer state compression (bitsandbytes-style) does not use a plain
linear grid — it uses a **dynamic exponent map**: 256 codes non-uniformly
spread so small magnitudes get fine resolution (like a floating-point
format) and the map always contains exactly `0.0` and `1.0`. Each tensor is
then quantized **blockwise**: split into fixed-size blocks, each block
normalized by its own absmax, and every normalized element snapped to the
*nearest* code in the shared 256-entry map.

### Building the 256-entry dynamic map

With `signed=True`, `max_exponent_bits=7`, `total_bits=8`:

$$
\text{non\_sign\_bits} = \text{total\_bits} - 1 = 7
$$

For each exponent index $i = 0, 1, \dots, 6$:

$$
\text{fraction\_items} = 2^{\,i} + 1, \qquad
\text{boundaries} = \mathrm{linspace}(0.1,\, 1.0,\, \text{fraction\_items})
$$

$$
\text{means}_j = \frac{\text{boundaries}_j + \text{boundaries}_{j+1}}{2}, \qquad
e = 10^{-(6-i)}
$$

Append every value $e \cdot \text{means}_j$ **and** every value
$-e \cdot \text{means}_j$ to the map. After all 7 exponent levels, append the
two exact values $0.0$ and $1.0$. Sort the resulting 256 values ascending —
this is the map $M \in \mathbb{R}^{256}$ (values span $[-1, 1]$).

### Blockwise absmax quantization

Split a 1-D array $v$ (Adam's non-negative second-moment state, flattened)
into consecutive blocks of size `blocksize` (the last block may be shorter).
For block $k$ with elements $v_k$:

$$
a_k = \max(|v_k|) \quad (\text{use } a_k = 1 \text{ if the block is all-zero})
$$

$$
\text{code}_i = \underset{c \in \{0,\dots,255\}}{\arg\min} \left| \frac{v_i}{a_k} - M_c \right|, \qquad
\hat{v}_i = M_{\text{code}_i} \cdot a_k
$$

## Task

Implement `quantize_dequantize_v_blockwise`:

```python
def quantize_dequantize_v_blockwise(v: np.ndarray, blocksize: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...
```

* `v` — 1-D non-negative float array (a flattened Adam `v` state).
* `blocksize` — block size for per-block absmax normalization.

Build the 256-entry dynamic map exactly as specified above, quantize `v`
blockwise against it, and return `(v_hat, codes, absmax)`:

* `v_hat` — dequantized reconstruction, same shape/length as `v`.
* `codes` — `uint8` array of per-element map indices, same length as `v`.
* `absmax` — `float32` array of per-block absmax values, length
  `ceil(len(v) / blocksize)`.

## Example

```python
import numpy as np
v = np.array([0.0, 0.5, 1.0, 3.0, 4.0])
v_hat, codes, absmax = quantize_dequantize_v_blockwise(v, blocksize=5)
# absmax = [4.0]  (single block, max(|v|) == 4.0)
# v_hat[i] ≈ v[i] within the map's resolution near 0 and near the block max
```

## What the gate checks

Gate **rel_err** rebuilds the same dynamic map and blockwise quantization
with a NumPy oracle and compares the global relative L2 error between your
`v_hat` and the oracle's, over several random `(v, blocksize)` instances.
