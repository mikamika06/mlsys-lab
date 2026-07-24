## Context

Weight-only quantization methods often search for a clipping value before quantizing weights. AWQ-style methods evaluate several clipping ratios and choose the one that gives the lowest reconstruction error.

For a weight matrix $W \in \mathbb{R}^{c \times n}$, each channel is clipped independently. For a candidate ratio $r$, the clipping bound for channel $i$ is

$$
m_i(r) = r \cdot \max_j |W_{ij}|.
$$

Values are clamped to the interval $[-m_i(r), m_i(r)]$. Symmetric $b$-bit quantization uses

$$
q_{\max} = 2^{b-1}-1,
$$

with scale

$$
s_i(r) = \frac{m_i(r)}{q_{\max}}.
$$

The reconstructed weights are

$$
\hat{W}_{ij}(r) =
\operatorname{round}\left(
\frac{\operatorname{clip}(W_{ij},-m_i(r),m_i(r))}
{s_i(r)}
\right)s_i(r).
$$

The search evaluates the mean squared error

$$
\mathrm{MSE}(r)=\frac{1}{cn}\sum_{i,j}(W_{ij}-\hat{W}_{ij}(r))^2
$$

for every candidate ratio and selects the ratio with the smallest error.

## Task

Implement `search_clip_ratio(W, ratios, bits)`.

The function receives:

- `W`: a 2-D NumPy array of floating point weights with shape `(channels, values_per_channel)`.
- `ratios`: a 1-D NumPy array of candidate clipping ratios.
- `bits`: the number of quantization bits.

Return:

```python
(best_index, mse_curve)
```

where `best_index` is the integer index of the ratio with the lowest MSE and `mse_curve` contains the MSE value for every ratio in the same order as `ratios`.

Compute the quantization search using NumPy operations. The returned curve must contain `float64` values.

## Example

```python
import numpy as np

W = np.array([
    [1.0, -2.0, 0.5],
    [3.0, -1.0, 2.0],
])

ratios = np.array([0.5, 0.75, 1.0])

index, curve = search_clip_ratio(W, ratios, 4)

# index is the position of the smallest value in curve
# curve contains one MSE value per candidate ratio
```

## What the gate checks

The gate computes an independent NumPy oracle that performs clipping, symmetric quantization, reconstruction, and MSE evaluation for every candidate ratio.

`argmin_index` checks that the selected index matches the oracle.

`mse_curve` checks that every value in the returned MSE curve matches the oracle curve numerically.
