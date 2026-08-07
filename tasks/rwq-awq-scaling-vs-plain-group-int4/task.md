## Context

Post-training quantization reduces memory by representing weights with low-bit integers. A common baseline is round-to-nearest group quantization. For a weight group $G$, symmetric int4 quantization uses a scale

$$s = \frac{\max(|G|)}{7},$$

then stores

$$q = \operatorname{clip}(\operatorname{round}(G/s), -8, 7),$$

and reconstructs the weights as $\hat{G}=qs$.

Activation-aware weight quantization (AWQ) adjusts weights before quantization so that channels important for the model output receive more precision. Given calibration activations $X$, channel importance is

$$i_j = \operatorname{mean}(|X_{:,j}|).$$

The AWQ channel multiplier is

$$a_j = \left(\frac{i_j}{\operatorname{mean}(i)}\right)^{1/2}.$$

The scaled weights are $W'_{:,j}=W_{:,j}a_j$. After group int4 quantization, the inverse scale restores the weights:

$$\hat{W}_{:,j}=\frac{\hat{W'}_{:,j}}{a_j}.$$

The layer output error is measured as

$$\mathrm{MSE}=\operatorname{mean}((XW^\top-X\hat{W}^\top)^2).$$

## Task

Implement `awq_vs_plain_group_int4_mse(W, X, group_size)`.

The function receives:

- `W`: a list of shape $(m,n)$ containing floating point weights.
- `X`: a list of shape $(b,n)$ containing calibration activations.
- `group_size`: the number of input channels in each quantization group.

Return a tuple:

```python
(awq_mse, plain_mse)
```

where both values are Python floats.

`plain_mse` must be computed from normal group int4 quantization of `W`. `awq_mse` must be computed after applying the AWQ channel scaling procedure, quantizing the scaled weights, and applying the inverse channel scaling.

Use deterministic Python operations.

## Example

```python

W = [[2.0, 0.5], [1.0, -3.0]]
X = [[4.0, 0.1], [3.0, 0.2]]

awq_mse, plain_mse = awq_vs_plain_group_int4_mse(W, X, 2)
```

The returned values are the two reconstruction errors for the same layer.

## What the gate checks

The gate computes a Python reference implementation of group int4 quantization and AWQ scaling. The returned values must match the reference within $10^{-6}$.

The generated calibration layers contain salient channels where the oracle verifies that

$$\mathrm{MSE}_{AWQ}<\mathrm{MSE}_{plain}.$$
