## Context

When large language models process tokens, activation vectors often exhibit heavy-tailed distributions: a few feature dimensions (called outliers) carry values orders of magnitude larger than the bulk. Quantizing the entire activation matrix with a single scale factor — naive per-tensor int8 — forces the quantization step to accommodate the outliers, wasting most of the 256 representable levels on an empty range far from zero.

For a quantizer with step size $\Delta$, the expected squared error for a uniformly distributed input is

$$\mathbb{E}[\epsilon^2] \approx \frac{\Delta^2}{12}\,.$$

A large $\Delta$ chosen to span the outliers inflates this error for every non-outlier element.

**LLM.int8() decomposition** (Dettmers et al., 2022) separates outlier columns:

1. Identify columns $j$ where $\max_i |X_{ij}| > \tau$ (the threshold).
2. Quantize the non-outlier columns with per-tensor int8, using a scale $s = \max |X_{\text{non-out}}| / 127$.
3. Represent outlier columns in `float16` (no int8 quantization step).

Reconstruct by combining both parts. The non-outlier sub-matrix gets a much smaller step size $\Delta$, dramatically reducing bulk MSE.

## Task

Implement:

```python
import numpy as np

def compare_quantization(X, threshold=6.0):
    """
    Return (mse_decomposition, mse_naive) — MSE of two int8 quantization
    strategies versus the float64 input X.
    """
    ...
```

**Naive per-tensor int8**: compute one scale $s = \max|X|\,/\,127$, quantize all elements with `np.round`, clip to $[-128, 127]$, cast to `int8`, dequantize by multiplying by $s$.

**Decomposition (LLM.int8()-style)**: columns whose max absolute value exceeds $\tau$ are "outlier" columns. Quantize the remaining columns with the same int8 scheme (scale computed from non-outlier values only). Represent outlier columns in `float16`. Reconstruct the full matrix by placing both parts back.

Return the mean squared error of each reconstruction versus the original `float64` $X$:

$$\text{MSE} = \frac{1}{nm} \sum_{i=1}^{n}\sum_{j=1}^{m} (X_{ij} - \hat{X}_{ij})^2\,.$$

Use NumPy only. Both returned values must be plain Python `float`.

## Example

```python
import numpy as np
X = np.array([[0.1, 50.0], [0.2, -60.0], [0.3, 45.0]])
mse_d, mse_n = compare_quantization(X, threshold=6.0)
# Column 1 is an outlier (max |val| = 60 > 6).
# mse_d < mse_n: decomposition keeps column 1 in fp16 while column 0
# gets a fine-grained int8 scale; naive uses one coarse scale for both.
```

## What the gate checks

Two gates on an outlier-heavy $64 \times 32$ fixture (columns 3, 17, 28 carry values scaled by $100\times$, $50\times$, $80\times$ respectively):

- **mse_within_tol**: both returned MSE values must agree with a NumPy oracle within $10^{-6}$.
- **decomp_wins**: the decomposition MSE must be strictly less than the naive MSE, confirming that separating outliers improves reconstruction.

The oracle recomputes both quantization strategies from scratch using the same algorithm, so passing requires implementing both strategies correctly.
