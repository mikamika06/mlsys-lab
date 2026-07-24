## Context

Affine quantization is a common technique to reduce the memory footprint of neural network weights while preserving inference accuracy.  
For a weight value $w$, an affine quantizer maps it to an integer code $q$ via

$$
q = \operatorname{round}\!\left(\frac{w - b}{s}\right),
$$

where $b$ is a bias (often the minimum of the group) and $s$ is a scale factor that spreads the dynamic range over the available integer values.  
The dequantized value is then recovered as

$$
\hat w = s\,q + b.
$$

When quantizing in *groups*, each contiguous block of weights shares the same bias and scale, allowing for efficient per-group calibration.

## Task

Implement `affine_group_quant_dequant(weights: np.ndarray, group_size: int = 64) -> Tuple[np.ndarray, np.ndarray]`.

* `weights` is a 2‑D NumPy array of shape $(N, D)$ containing floating‑point weights.
* The function should process the rows in consecutive groups of size `group_size`.  
  For each group:
  * Compute `bias = min(group)` and
    $$ s = \frac{\max(\text{group}) - \text{min}(\text{group})}{255}. $$
    If all values are equal, set `s = 1.0` to avoid division by zero.
  * Quantize each weight in the group with the formula above and **clamp** the result to the signed 8‑bit range $[-128,127]$.
  * Dequantize using the same bias and scale.

The function must return a tuple `(q_codes, recon)` where

* `q_codes` is an integer array of dtype `np.int8` containing the quantized codes,
* `recon` is a floating‑point array (dtype `float64`) containing the dequantized weights.

Both outputs should have the same shape as `weights`.

## Example

```python
import numpy as np
from your_module import affine_group_quant_dequant

A = np.array([[0.1, 0.2],
              [0.3, 0.4],
              [0.5, 0.6],
              [0.7, 0.8]], dtype=np.float64)

q_codes, recon = affine_group_quant_dequant(A, group_size=2)
print(q_codes)
# [[-128 -128]
#  [-127 -127]
#  [-126 -126]
#  [-125 -125]]

print(recon)
# [[0.   0. ]
#  [0.01 0.01]
#  [0.02 0.02]
#  [0.03 0.03]]
```

## What the gate checks

The grader compares your integer codes to a reference implementation that follows the exact algorithm described above.  
It also verifies that the reconstructed weights satisfy

$$
\max_{i,j} |\hat w_{ij} - w_{ij}| \leq 10^{-6}.
$$

If both conditions hold for all test cases, the gate `exact_match` passes with a score of `1.0`. Otherwise it fails.
