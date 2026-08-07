## Context

FP8 E4M3 is a floating point format commonly used for neural network inference and training. A tensor is transformed before quantization by dividing by a scale $s$:

$$
q = \mathrm{FP8Quantize}\left(\frac{x}{s}\right),
$$

and reconstructed with

$$
\hat{x} = s \cdot q.
$$

The E4M3 format has a maximum finite representable value of $448$. A common initialization is

$$
s_0 = \frac{\max(|x|)}{448},
$$

but this assumes the largest values should always be preserved. A slightly smaller or larger scale can improve the total error by balancing clipping error and rounding error.

For a candidate scale $s_i$, the mean squared error is

$$
\mathrm{MSE}(s_i) = \frac{1}{n}\sum_{j=1}^{n}(x_j-\hat{x}_{i,j})^2 .
$$

A production quantizer can search a small grid of scales around $s_0$ and select the scale index with the smallest MSE.

## Task

Implement `search_fp8_scale(x)`.

The function receives a list of floats of floating point values and must return:

```python
best_index, scales, mses
```

where:

- `scales` is a list of candidate scales.
- `mses[i]` is the reconstruction MSE when quantizing with `scales[i]`.
- `best_index` is the integer index of the minimum value in `mses`.

Use the following scale grid:

$$
s_i = s_0 \cdot 2^{(i-4)/8}, \quad i=0,\dots,8 .
$$

For each scale, quantize $x/s_i$ to E4M3, clip values outside the finite range, round to the nearest representable E4M3 value, then dequantize by multiplying by $s_i$.

The implementation should return the complete MSE curve, not only the best scale.

## Example

```python

x = [1.0, 20.0, 100.0]

index, scales, mses = search_fp8_scale(x)

# index is the position of the lowest MSE candidate
# scales contains 9 searched scale values
# mses contains the reconstruction error for every scale
```

## What the gate checks

The gate computes an independent Python oracle for E4M3 quantization and the same scale grid.

`argmin_index` checks that the returned index matches the oracle's minimum-MSE scale.

`mse_curve_error` checks that the submitted MSE curve matches the oracle curve within a small numerical tolerance. A solution that only chooses a scale without evaluating the clipping and rounding tradeoff will fail.
