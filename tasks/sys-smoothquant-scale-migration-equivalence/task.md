## Context

SmoothQuant is an inference quantization technique that reduces activation outliers by migrating their scale into the weights while keeping the floating point computation unchanged.

For a weight matrix $W \in \mathbb{R}^{m \times d}$ and activation matrix $X \in \mathbb{R}^{n \times d}$, the original matrix multiplication is

$$Y = W X^\top.$$

A positive per-channel migration vector $s \in \mathbb{R}^{d}$ transforms the tensors as

$$W'_{ij} = W_{ij}s_j,$$

$$X'_{kj} = \frac{X_{kj}}{s_j}.$$

The channel scales cancel during multiplication:

$$W'X'^\top = WX^\top.$$

The benefit appears after quantization because activation ranges can become smaller. A symmetric int8 quantizer can be described as

$$q(z)=\operatorname{round}\left(\frac{z}{a}\right), \qquad \hat{z}=a q(z),$$

where

$$a=\frac{\max(|z|)}{127}.$$

## Task

Implement `smoothquant_migrate(W, X, s)`.

The function receives:

- `W`: a NumPy array with shape $(m,d)$.
- `X`: a NumPy array with shape $(n,d)$.
- `s`: a one-dimensional NumPy array with length $d$ containing positive channel scales.

Return `(W_migrated, X_migrated)`.

The returned arrays must satisfy:

- `W_migrated` multiplies each input channel of `W` by the matching value in `s`.
- `X_migrated` divides each input channel of `X` by the matching value in `s`.
- `W_migrated @ X_migrated.T` is numerically equivalent to `W @ X.T`.

Use NumPy broadcasting rather than manually looping through channels.

## Example

```python
import numpy as np

W = np.array([[2.0, 4.0], [1.0, -3.0]])
X = np.array([[6.0, 1.0], [2.0, 8.0]])
s = np.array([3.0, 0.5])

W2, X2 = smoothquant_migrate(W, X, s)

# W2:
# [[ 6.0,  2.0],
#  [ 3.0, -1.5]]

# X2:
# [[ 2.0,  2.0],
#  [ 0.666666..., 16.0]]
```

## What the gate checks

The gate builds a NumPy oracle implementation of the migration and tests several tensors with activation outliers.

The floating point migration error is measured as

$$\mathrm{rel\_err}=\frac{\lVert W'X'^\top-WX^\top\rVert}{\lVert WX^\top\rVert+10^{-12}}.$$

The result must satisfy $\mathrm{rel\_err} \le 10^{-6}$.

The gate also quantizes weights and activations with a NumPy symmetric int8 quantizer. The quantized matrix multiplication error after migration must be lower than the original quantized error, so the reported

$$\mathrm{quant\_error\_ratio}=\frac{\mathrm{error\ after\ migration}}{\mathrm{error\ before\ migration}}$$

must be less than $1$.
