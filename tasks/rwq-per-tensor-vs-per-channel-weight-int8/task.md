## Context

Symmetric int8 quantization of a weight matrix $W \in \mathbb{R}^{r\times c}$
maps each value onto $\{-127,\dots,127\}$ with a single affine scale:

$$
\text{scale} = \frac{\max(|g|)}{127}, \qquad
\hat g_i = \text{scale}\cdot\mathrm{clip}\!\big(\mathrm{round}(g_i/\text{scale}),\, -127,\, 127\big)
$$

The only design choice is *what "$g$" is* — i.e. which elements share
one scale:

- **Per-tensor**: $g = W$ (flattened) — one scale for the whole matrix.
- **Per-channel**: $g = W[i,:]$ for each output row $i$ — one scale
  *per row*.

If different rows have very different magnitudes (extremely common in
real weight matrices), a single per-tensor scale is dragged wide by the
loudest row, crushing every quiet row's rounding resolution. A
per-channel scale lets each row use the full 8-bit grid on its own
range instead.

## Task

Implement `int8_mse_per_tensor_vs_per_channel`:

```python
def int8_mse_per_tensor_vs_per_channel(W: np.ndarray) -> tuple[float, float]:
    ...
```

- `W`: `(out_features, in_features)` `float64` weight matrix.

1. **Per-tensor**: quantize the entire matrix with one scale
   (`scale = max(|W|) / 127`, formula above), reconstruct $\hat W_{\text{pt}}$,
   compute `mse_per_tensor = mean((W_hat_pt - W) ** 2)` over *all*
   elements.
2. **Per-channel**: quantize each row $W[i,:]$ independently with its
   own scale (`scale_i = max(|W[i,:]|) / 127`), reconstruct
   $\hat W_{\text{pc}}$, compute `mse_per_channel = mean((W_hat_pc - W) ** 2)`
   over all elements.

Return `(mse_per_tensor, mse_per_channel)`.

## Example

```python
import numpy as np
W = np.vstack([
    np.full(8, 100.0),   # one huge row
    np.full(8, 0.5),     # one tiny row
])
mse_pt, mse_pc = int8_mse_per_tensor_vs_per_channel(W)
# per-tensor: scale = 100/127 ~ 0.79 -- the tiny row (0.5) rounds to the
#   nearest multiple of 0.79, i.e. to 0 or 0.79: huge relative error.
# per-channel: the tiny row gets its own scale = 0.5/127, reconstructing
#   it almost exactly. So mse_pc << mse_pt.
```

## What the gate checks

The grader builds several seeded weight matrices whose rows span a wide
range of magnitudes and computes both MSEs independently in NumPy with
the exact formulas above.

`per_tensor_abs_err` / `per_channel_abs_err` are the worst-case absolute
difference between your two returned MSEs and the oracle's across all
cases (each must be `<= 1e-9`) — this is deterministic floating-point
arithmetic, so any real formula mismatch (wrong scale, wrong axis,
missing clip) produces an error far above tolerance.
`order_margin` is `mse_per_tensor - mse_per_channel` from *your own*
returned values (must be `>= 0`) — every case is built so per-channel
quantization is genuinely no worse than per-tensor; a solution that
silently swaps the two or computes only one real value fails this even
if it happens to pass the two MSE checks in isolation.
