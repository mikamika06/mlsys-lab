## Context

For a linear layer $Y = XW^\top$ with $W \in \mathbb{R}^{m\times n}$,
$X \in \mathbb{R}^{b\times n}$, plain **round-to-nearest (RTN)**
per-output-row INT4 quantization of $W$ uses the same 16-level grid for
every input channel of that row:

$$
\Delta_i = \frac{\max_j |W_{i,j}|}{7}, \qquad
\hat W_{i,j} = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{W_{i,j}}{\Delta_i}\right), -8, 7\right)\Delta_i .
$$

If a few input channels $j$ carry activations $X_{:,j}$ far larger in
magnitude than the rest (**salient channels**), their contribution to $Y$
is $X_{:,j}\hat W_{i,j}$ — the same absolute rounding error on
$\hat W_{i,j}$ gets amplified by that large activation, dominating the
output error, while ordinary channels' quantization is comparatively
harmless.

**AWQ** (Activation-aware Weight Quantization) protects salient channels
by rescaling *before* quantizing: compute a per-input-channel scale
$s_j$ from the activations, scale $W_{:,j} \leftarrow W_{:,j}\cdot s_j$
(so a salient channel's weights become larger relative to the row's max,
claiming more of the INT4 grid's resolution), quantize with the same
formula, then divide back out by $s_j$ to reconstruct
$\hat W^{\text{AWQ}}$. A simple, standard choice of scale is the
per-channel mean activation magnitude:

$$
s_j = \frac{1}{b}\sum_{k} |X_{k,j}| .
$$

## Task

Implement `compare_awq_rtn_error(W, X)`:

```python
def compare_awq_rtn_error(W: np.ndarray, X: np.ndarray) -> tuple[float, float, float]:
    ...
```

- `W`: `(out_dim, in_dim)` float weight matrix.
- `X`: `(batch, in_dim)` float activations.

1. Compute `Y_true = X @ W.T`.
2. **RTN**: quantize `W` directly with the per-row INT4 formula above;
   `err_rtn` = relative Frobenius-norm error of `X @ W_hat_rtn.T` vs
   `Y_true`, i.e. `||Y_approx - Y_true||_F / ||Y_true||_F`.
3. **AWQ**: compute `s_j = mean_k |X[k, j]|` for every input channel;
   quantize `W * s` (broadcast over columns) with the same INT4 formula,
   then divide the quantized result by `s` to get `W_hat_awq`; `err_awq`
   is the same relative Frobenius error using `W_hat_awq`.
4. `reduction = 1 - err_awq / err_rtn`.

Return `(err_rtn, err_awq, reduction)`.

## Example

```python
import numpy as np

rng = np.random.default_rng(0)
W = rng.normal(size=(8, 16))
X = rng.normal(size=(32, 16))
X[:, [2, 9]] *= 100.0   # channels 2 and 9 are salient

err_rtn, err_awq, reduction = compare_awq_rtn_error(W, X)
# err_awq is noticeably smaller than err_rtn; reduction > 0
```

## What the gate checks

The gate loads a fixed fixture (`W.npy`, `X.npy`: a 16x32 weight and
64x32 activations with three channels scaled 120x, deliberately salient)
plus several seeded synthetic `(W, X)` pairs with different salient
channels and scales. For each, the oracle independently recomputes
`(err_rtn, err_awq, reduction)` with the exact formulas above.

Your returned triple is compared to the oracle's with the `rel_err`
scorer (relative L2 error over the 3-vector), and the worst case across
every scenario must be `< 1e-6`. Using the wrong AWQ scale formula (e.g.
scaling by `1/s` instead of `s`, or forgetting to divide back out after
quantizing), computing RTN and AWQ against the wrong `Y_true`, or getting
the relative-error or reduction formula backwards will all miss the
tolerance.
