## Context

Symmetric INT8 quantization of a weight matrix $W$ maps values to the
signed 8-bit range $[-127,127]$ with a single scale $s$:

$$
q = \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{W}{s}\right), -127, 127\right),
\qquad \hat W = q\,s .
$$

The scale is fit to the data's magnitude: $s = \max(|W|)/127$.

The granularity at which that scale is computed matters:

- **Per-tensor**: one scale for the entire matrix,
  $s_{\text{tensor}} = \max_{i,j}(|W_{ij}|)/127$.
- **Per-channel** (per output row): one scale per row,
  $s_i = \max_j(|W_{ij}|)/127$.

When rows have very different magnitude ranges, a single per-tensor scale
is sized for the loudest row, and every quieter row gets crushed toward
zero and loses precision. A per-row scale gives each row the full
resolution INT8 can offer for *its own* range. Reconstruction quality is
measured with mean squared error,

$$
\operatorname{MSE}(W,\hat W) = \frac{1}{\lvert W\rvert}\sum_{i,j}(W_{ij}-\hat W_{ij})^2 .
$$

## Task

Implement `quant_granularity_errors`:

```python
def quant_granularity_errors(W: np.ndarray) -> dict:
    ...
```

- `W`: a `(rows, cols)` float64 weight matrix.

Quantize `W` with symmetric INT8 two ways — once with a single per-tensor
scale, once with one scale per row (per-channel) — and return a `dict`:

```python
{
    "mse_per_tensor": float,   # MSE(W, W reconstructed per-tensor)
    "mse_per_channel": float,  # MSE(W, W reconstructed per-channel)
    "winner": "per_tensor" | "per_channel",  # whichever has the lower MSE
}
```

`"winner"` must be the string with the strictly lower MSE (ties broken
toward `"per_channel"`, matching the grader).

## Example

```python
import numpy as np

W = np.array([
    [100.0, -100.0, 50.0],
    [0.02, -0.01, 0.015],
])
out = quant_granularity_errors(W)
# out["mse_per_tensor"] is large: the tiny second row is scaled by the
# same s as the huge first row and rounds almost entirely to 0.
# out["mse_per_channel"] is much smaller: the second row gets its own
# scale sized for its own tiny magnitudes.
# out["winner"] == "per_channel"
```

## What the gate checks

The grader builds several weight matrices from a seeded NumPy generator
— rows with wildly different magnitude spreads, rows with a uniform
shared range, and a matrix with one dominant outlier row — and computes
`mse_per_tensor`, `mse_per_channel`, and the true `winner` independently
in NumPy from the quantization formulas above, never calling your
function.

Two metrics are checked:

* **rel_err** — the relative L2 error between your `[mse_per_tensor,
  mse_per_channel]` pair and the oracle's, worst-case over all
  scenarios. Must satisfy $\text{rel\_err} \le 10^{-6}$.
* **winner_correct** — whether your reported `"winner"` matches the
  oracle's winner on every scenario. Must equal `1.0`.

Computing both reconstructions with the same (per-tensor) scale,
computing the per-channel scale along the wrong axis, forgetting the
clip to $[-127,127]$, or reporting the winner backwards will all be
caught.
