## Context

INT8 W8A8 quantization casts both weights and activations to INT8. Real
LLM activations often have a handful of **outlier channels** with much
larger magnitude than the rest — a single per-tensor INT8 scale sized to
cover those outliers wastes almost all of its precision on the common
case. Weights, by contrast, are usually comparatively well-behaved.
SmoothQuant exploits the fact that a linear layer's output is invariant
to a **per-input-channel** rescaling split between activation and
weight:

$$
X W^\top = \Big(\frac{X}{s}\Big)\big(W \cdot s\big)^\top,
\qquad
s_j = \frac{\max_i |X_{i,j}|^{\alpha}}{\max_o |W_{o,j}|^{1-\alpha}}
$$

applied per input channel $j$ (dividing column $j$ of $X$, multiplying
column $j$ of $W$). The migration itself is exact — no error is
introduced by the rescaling alone. It only matters once BOTH tensors are
subsequently cast to INT8: moving dynamic range off the outlier-heavy
activations and onto the (now slightly less well-behaved, but still
much better) weights lowers the *combined* quantization error.

## Task

Implement `smoothquant_w8a8_comparison`:

```python
def smoothquant_w8a8_comparison(X: np.ndarray, W: np.ndarray, alpha: float) -> dict:
    ...
```

- `X`: `(n, d_in)` activations.
- `W`: `(d_out, d_in)` weight matrix, `Y = X @ W.T`.
- `alpha`: float in `(0, 1)`, the migration strength.

Using **per-tensor symmetric INT8** quantization (`scale = max(|x|) /
127`, `q = clip(round(x / scale), -127, 127)`, dequantize `q * scale`)
applied independently to each tensor:

1. Quantize `X` and `W` directly (no migration); compute
   `error_raw` = Frobenius relative error of the resulting `Y` vs the
   exact FP `X @ W.T`.
2. Compute the per-channel scale `s` above, migrate
   `X_smooth = X / s`, `W_smooth = W * s`, quantize both, and compute
   `error_smoothed` the same way.
3. `improvement_ratio = error_smoothed / error_raw`.

Return `{"error_raw": float, "error_smoothed": float, "improvement_ratio": float}`.

## Example

```python
import numpy as np

rng = np.random.default_rng(0)
X = rng.standard_normal((50, 10)) * 0.5
X[:, 0] *= 20.0   # one large outlier channel
W = rng.standard_normal((8, 10)) * 0.3

out = smoothquant_w8a8_comparison(X, W, alpha=0.5)
# out["improvement_ratio"] < 1.0 -- smoothing measurably reduces error
# on activations with a real outlier channel.
```

## What the gate checks

The grader loads a committed `X.npy`/`W.npy` fixture (three genuine
outlier channels, 10-30x the typical magnitude) graded at `alpha=0.5`,
plus several additional seeded synthetic `(X, W, alpha)` triples, and
computes all three values independently in NumPy with the same
per-tensor symmetric INT8 round-trip — never calling your function,
never hardcoding an expected value.

`rel_err` is `scorers.rel_err` applied to the 3-vector `[error_raw,
error_smoothed, improvement_ratio]` against the oracle's, taking the
worst case across all cases, and must be `<= 1e-6`. Migrating per-tensor
instead of per-channel, swapping which tensor gets divided vs
multiplied by `s`, or using the wrong INT8 rounding convention will all
shift the reported errors and their ratio.
