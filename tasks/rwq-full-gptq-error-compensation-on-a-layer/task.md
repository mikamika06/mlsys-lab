## Context

Plain round-to-nearest (RTN) quantization rounds every weight independently
and throws the rounding error away. **GPTQ** instead treats quantization as
a sequential, layer-aware process: it quantizes one column of the weight
matrix at a time and immediately pushes that column's rounding error onto
the *not-yet-quantized* columns, weighted by the inverse of the layer's
calibration Hessian $H = X^\top X \in \mathbb{R}^{d_{in}\times d_{in}}$ ($X$
being calibration activations). Because $H^{-1}$ encodes how columns
co-vary, this compensation genuinely reduces the layer's *output* error
compared to RTN — it is the core trick behind GPTQ-style post-training
quantization used in production LLM quantizers.

This task asks for the **full column-by-column GPTQ update** in natural
(left-to-right) column order — no importance reordering (that is a
separate technique, "act-order").

### Algorithm

Given $W \in \mathbb{R}^{d_{out}\times d_{in}}$, calibration activations
$X \in \mathbb{R}^{n_{cal}\times d_{in}}$, bit width $b$, and relative
dampening $\delta$:

1. **Hessian**: $H = X^\top X$.
2. **Dampen and invert**: $H \leftarrow H + \delta\cdot\overline{\mathrm{diag}(H)}\cdot I$,
   then $H^{-1}$.
3. **Per-column quantization params** (computed once, from the *original*
   $W$, before any updates): for column $c$, with $m=\min(0,\min W_{:,c})$,
   $M=\max(0,\max W_{:,c})$,
   $$
   s_c = \frac{M-m}{2^{b}-1}\ (\text{or } 1 \text{ if } M=m), \qquad
   z_c = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{-m}{s_c}\right), 0, 2^{b}-1\right)
   $$
   and $\mathrm{quant}(w) = \left(\mathrm{clip}(\mathrm{round}(w/s_c)+z_c,\,0,\,2^{b}-1) - z_c\right) s_c$.
4. **Sequential update**, $i = 0,\dots,d_{in}-1$ in natural order:
   $$
   q = \mathrm{quant}_{i}(W_{:,i}), \qquad
   e = \frac{W_{:,i} - q}{(H^{-1})_{ii}}
   $$
   $$
   W_{:,i} \leftarrow q, \qquad
   W_{:,i+1:} \leftarrow W_{:,i+1:} - e\,(H^{-1})_{i,\,i+1:}
   $$
5. **Layer output error**: with $\hat W$ the resulting quantized matrix and
   $W$ the *original* matrix,
   $$
   \mathrm{mse} = \frac{1}{n_{cal}\,d_{out}}\left\lVert X\hat W^\top - XW^\top\right\rVert_F^2 .
   $$

## Task

Implement `gptq_quantize_layer`:

```python
def gptq_quantize_layer(W: np.ndarray, X: np.ndarray, nbits: int, damp: float) -> tuple[np.ndarray, float]:
    ...
```

* `W` — `(d_out, d_in)` weight matrix.
* `X` — `(n_cal, d_in)` calibration activation matrix.
* `nbits` — bits per column (unsigned affine quantization, as above).
* `damp` — relative Hessian dampening $\delta$.

Return `(Wq, mse)`:

* `Wq` — `(d_out, d_in)` quantized weight matrix, produced by running the
  algorithm above in **natural column order** $0, 1, \dots, d_{in}-1$.
* `mse` — `float`, the layer output MSE as defined in step 5 above (using
  the *original*, unquantized `W` for $XW^\top$).

## Example

```python
import numpy as np
rng = np.random.default_rng(0)
X = rng.normal(size=(40, 6))
W = rng.normal(size=(4, 6))
Wq, mse = gptq_quantize_layer(W, X, nbits=4, damp=0.01)
# Wq.shape == W.shape; mse >= 0.0
```

## What the gate checks

* **wq_max_abs_err** — the max-abs difference between your `Wq` and the
  matrix produced by re-running the exact algorithm above (natural order,
  same dampening/quant formulas) with a NumPy oracle on the fixed
  calibration fixtures.
* **mse_rel_err** — the relative difference between your reported `mse`
  and the oracle's `mse` on the same run.
* **beats_rtn** — your `mse` must be strictly lower than the MSE of a
  plain RTN baseline (each column quantized independently with the same
  per-column scale/zero-point, no error compensation) on the same
  fixtures — proof that the H^-1 error compensation is actually doing its
  job.
