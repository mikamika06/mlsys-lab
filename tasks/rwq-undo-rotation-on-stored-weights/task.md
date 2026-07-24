## Context

In neural network quantization, weights are often compressed from floating-point to low-precision integers. A common technique to reduce quantization error is *rotation pre-quantization*: apply an orthogonal matrix $H \in \mathbb{R}^{m \times m}$ to the weight matrix $W \in \mathbb{R}^{m \times n}$ before quantization,

$$W_{\text{rot}} = H W.$$

The quantized representation stores $Q(W_{\text{rot}}) = \operatorname{round}(W_{\text{rot}} / s) + z$, where $s$ is the scale factor and $z$ is the zero-point. At inference time the original weights are recovered by

$$\widehat{W} = H^\top \bigl( (Q(W_{\text{rot}}) - z) \cdot s \bigr).$$

This reconstruction operation must be implemented correctly for deployment.

## Task

Implement the function

```python
def reconstruct_weights(W_quantized: np.ndarray, H: np.ndarray, scale: float, zero_point: float) -> np.ndarray:
    ...
```

**Inputs**

- `W_quantized`: integer array of shape $(m, n)$ containing the quantized rotated weights (dtype `np.int8`).
- `H`: orthogonal matrix of shape $(m, m)$ used for the rotation.
- `scale`: the scale coefficient (positive float).
- `zero_point`: the zero-point (integer cast to float).

**Output**

- A float64 array of shape $(m, n)$ approximating the original weight matrix $W$.

The reconstruction formula:

1. Dequantize: $W_{\text{dequant}} = (W_{\text{quantized}} - \text{zero\_point}) \cdot \text{scale}$.
2. Inverse rotate: $W_{\text{recovered}} = H^\top \, W_{\text{dequant}}$.

Use vectorized NumPy operations; no explicit loops.

## Example

```python
import numpy as np

H = np.array([[0.6, 0.8],
              [0.8, -0.6]])  # orthogonal 2×2

W_quantized = np.array([[12,  5],
                         [ 1, 20]], dtype=np.int8)
scale = 0.5
zero_point = 0.0

W_rec = reconstruct_weights(W_quantized, H, scale, zero_point)
# Step 1: dequantize → [[6., 2.5], [0.5, 10.]]
# Step 2: H^T @ dequant → [[ 3.8,  7. ],
#                           [ 8.2, -8. ]]
# (values approximate)
```

## What the gate checks

The gate compares your output against an oracle that runs the same dequantisation and inverse rotation using `float64` arithmetic. The metric is the maximum absolute element‑wise error:

$$\text{max\_abs\_err} = \max_{i,j} |\widehat{W}_{ij} - W_{\text{oracle},ij}|.$$

You must achieve $\text{max\_abs\_err} \le 10^{-12}$ to pass. Any difference larger than machine epsilon usually indicates a mistake in the formula (e.g., transposing $H$ incorrectly, forgetting the zero‑point, or using the wrong order of operations).

This task expects you to write the reconstruction logic yourself; using an external library that already implements the pipeline would still produce the correct result, but the gate checks the functional correctness of the output array.
