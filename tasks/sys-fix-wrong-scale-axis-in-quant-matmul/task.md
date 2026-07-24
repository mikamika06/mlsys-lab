## Context

In quantized inference, weight matrices are stored as low-bit integer codes with
per-channel scales to reduce memory footprint. For a weight matrix
$W \in \mathbb{R}^{K \times N}$ quantized with per-output-channel scales
$s \in \mathbb{R}^{N}$, dequantization computes

$$W_{ij} = \mathrm{codes}_{ij} \cdot s_j .$$

In NumPy, a $(K, N)$ code matrix times a length-$N$ scale vector broadcasts
naturally: each $s_j$ multiplies column $j$. Equivalently one can write
`scales[np.newaxis, :]` for shape $(1, N)$.

A common mistake is to reshape the scale vector along the wrong axis — writing
`scales[:, np.newaxis]` (shape $(N, 1)$) instead of `scales` or
`scales[np.newaxis, :]` (shape $(1, N)$). This applies $s_i$ to *row* $i$ instead
of *column* $j$, producing a silently wrong result with no shape error.

## Task

The file `starter.py` contains a buggy `quant_matmul(x, codes, scales)`. It should
dequantize int8 weight codes using per-output-channel scales and return the matrix
product with input activations. Find and fix the broadcasting bug. Do not change the
function signature.

## Example

```python
import numpy as np
x = np.array([[1.0, 2.0],
              [3.0, 4.0]])                      # (2, 2)
codes = np.array([[10, -5],
                  [20, 30]], dtype=np.int8)     # (2, 2)
scales = np.array([0.1, 0.01])                  # per-output-channel, (2,)

# Correct dequant: W = [[10*0.1, -5*0.01],     = [[ 1.0, -0.05],
#                        [20*0.1, 30*0.01]]        [ 2.0,  0.30]]
# result = x @ W
```

## What the gate checks

A single gate reports the maximum relative error $\mathrm{rel\_err}$ across five
test cases of varying dimensions:

$$\mathrm{rel\_err}
  = \frac{\lVert \text{got} - \text{ref} \rVert_2}
         {\lVert \text{ref} \rVert_2 + 10^{-12}}
  < 10^{-3} .$$

The reference is computed with the correct formula
$W = \mathrm{codes} \times s$ using standard NumPy broadcasting. The broken
starter applies the scale on the wrong axis, giving
$\mathrm{rel\_err} \gg 10^{-3}$.
