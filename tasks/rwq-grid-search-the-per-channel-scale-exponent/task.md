## Context

**AWQ** (Activation-aware Weight Quantization) observes that a weight
matrix's most important *columns* (input channels) are the ones multiplied
by the largest-magnitude activations, and protects them by rescaling before
quantization: multiply a channel's weights up by $s_i$ and divide the
corresponding activations down by $s_i$, so the product `W @ X` is
unchanged in full precision, but the up-scaled weight column now uses more
of the quantization grid's dynamic range and rounds more accurately.

Rather than hand-picking $s_i$, AWQ derives it from the average activation
magnitude of channel $i$, $s_{x}[i] = \text{mean}_k |X_{ki}|$, raised to a
single scalar power $r$ (the "ratio"):

$$
s[i] = s_x[i]^{\,r}, \qquad
s \leftarrow \frac{s}{\sqrt{\max_i s[i] \cdot \min_i s[i]}}
$$

($r=0$ gives no scaling at all; $r=1$ scales every channel exactly
proportional to its average activation magnitude; the normalization keeps
the scale vector's own dynamic range balanced around 1.) The best $r$ is
found by a small **grid search**: for each candidate ratio, scale $W$ by
$s$, quantize the scaled weights, undo the scaling, and measure how much
the *output* changed on real calibration activations — then keep the ratio
with the smallest output MSE.

Given weights $W \in \mathbb{R}^{M \times K}$ and calibration activations
$X \in \mathbb{R}^{N \times K}$ ($N$ calibration samples, $K$ input
channels), and a fixed grid of ratios
$\text{RATIOS} = (0.0, 0.1, 0.2, \dots, 1.0)$:

For each ratio $r \in \text{RATIOS}$:

$$
s = s_x^{\,r} \big/ \sqrt{\max(s_x^{\,r})\cdot\min(s_x^{\,r})}, \qquad
W_{\text{sc}} = W \odot s \; (\text{broadcast over columns})
$$

Quantize $W_{\text{sc}}$ **per output row** (each row of $M$ gets its own
symmetric scale) to `n_bits` with round-to-nearest, dequantize back to
$\hat W_{\text{sc}}$, then undo the AWQ scaling: $\hat W = \hat
W_{\text{sc}} / s$. The output MSE for this ratio is

$$
\mathrm{mse}(r) = \frac{1}{NM}\left\lVert X W^\top - X \hat W^\top \right\rVert_F^2 .
$$

## Task

Implement `awq_ratio_search`:

```python
def awq_ratio_search(W: np.ndarray, X: np.ndarray, n_bits: int = 4) -> tuple[int, float]:
    ...
```

* `W` — 2‑D array of shape $(M, K)$, the weight matrix.
* `X` — 2‑D array of shape $(N, K)$, calibration activations.
* `n_bits` — bit width used for the per-row symmetric round-to-nearest
  weight quantization (guard against an all-zero row by using scale $1.0$
  there).

Use the fixed grid `RATIOS = (0.0, 0.1, ..., 1.0)` (11 points, in that
order). For each ratio, compute $s$, $\hat W$ and $\mathrm{mse}(r)$ exactly
as described above (guard against an all-zero activation channel by
flooring $s_x$ at a tiny epsilon before taking the power). Return
`(best_index, best_mse)`: the index into `RATIOS` of the ratio with the
smallest MSE, and that MSE value as a plain float. Use vectorised NumPy —
a Python loop over the 11 grid points is fine (it is a fixed, tiny grid),
but avoid looping over matrix elements.

## Example

```python
import numpy as np
W = np.random.default_rng(1).normal(size=(4, 6)).astype(np.float32)
X = np.random.default_rng(2).normal(size=(20, 6)).astype(np.float32)
X[:, 0] *= 10.0  # one salient/outlier channel

idx, mse = awq_ratio_search(W, X, n_bits=4)
print(idx, mse)   # idx in [0, 10]; mse >= 0.0
```

## What the gate checks

Two gates against a random weight matrix and a batch of calibration
activations with two salient outlier channels:

- **argmin_exact** — your returned ratio index must exactly equal the
  oracle's argmin over the same fixed 11-point grid.
- **mse_rel_err** — the MSE value you return at that index must match the
  oracle's MSE (computed the same way: quantize → dequantize → undo scale
  → compare outputs on the calibration activations) to within relative
  error $10^{-6}$.
