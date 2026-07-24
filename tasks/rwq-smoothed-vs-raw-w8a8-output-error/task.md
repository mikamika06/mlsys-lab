## Context

**SmoothQuant** migrates quantization difficulty from activations to weights by
applying a per-channel smoothing factor $s \in \mathbb{R}^{n}_{>0}$:

$$\hat{X} = X \cdot \text{diag}(s)^{-1}, \quad \hat{W} = \text{diag}(s) \cdot W$$

so the product $\hat{X} \hat{W} = X W$ is preserved exactly in floating point.
After migration, both $\hat{X}$ and $\hat{W}$ are quantized to int8 with
per-tensor symmetric quantization:

$$Q_{\text{sym}}(T, s_T) = \text{clip}\!\left(\text{round}\!\left(\frac{T}{s_T}\right), -127, 127\right)$$

where $s_T = \frac{\max|T|}{127}$.

**Raw W8A8** quantizes $X$ and $W$ directly without smoothing.
**Smoothed W8A8** quantizes $\hat{X}$ and $\hat{W}$.

The MSE of the reconstructed output is:

$$\text{MSE} = \frac{1}{mn}\| Y_{\text{quant}} - XW \|_F^2$$

where $Y_{\text{quant}} = \text{dequant}(\text{quant}(X_\text{in})) \cdot \text{dequant}(\text{quant}(W_\text{in}))$
with $X_\text{in}, W_\text{in}$ being either raw or smoothed versions.

## Task

Implement `w8a8_output_errors(X, W, s)`:

```python
def w8a8_output_errors(X, W, s):
    ...
```

- `X`: float32 array of shape `(m, n)` — input activations.
- `W`: float32 array of shape `(n, k)` — weight matrix.
- `s`: float32 array of shape `(n,)` — per-channel smoothing factors.

Return a tuple `(mse_raw, mse_smoothed)` where:
- `mse_raw`: MSE of the W8A8 output using raw $X$ and $W$ (no smoothing).
- `mse_smoothed`: MSE of the W8A8 output after SmoothQuant migration with $s$.

Use per-tensor symmetric int8 quantization (scale = max|T|/127, clip to [-127, 127]).

## Example

```python
import numpy as np
rng = np.random.default_rng(0)
X = rng.normal(0, 1, (4, 8)).astype(np.float32)
W = rng.normal(0, 1, (8, 4)).astype(np.float32)
s = np.abs(X).max(axis=0) ** 0.5 + 1e-6  # typical SmoothQuant factor
mse_raw, mse_smoothed = w8a8_output_errors(X, W, s)
# mse_smoothed <= mse_raw in the typical case
```

## What the gate checks

The grader computes both MSEs using the same per-tensor symmetric int8
quantization oracle, verifies that each is within $10^{-6}$ of the reference,
and checks that `mse_smoothed <= mse_raw` (smoothing should not increase error).
The gate passes when both conditions hold.
