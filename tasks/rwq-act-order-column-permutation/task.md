## Context

GPTQ quantizes a weight matrix one column at a time, compensating each
column's rounding error onto the *not-yet-quantized* columns using the
inverse of the layer's Hessian $H \in \mathbb{R}^{d_{in}\times d_{in}}$
(estimated from calibration activations as $H = X^\top X$). Quantizing
columns in their natural left-to-right order wastes precision on columns
that barely matter and starves columns the Hessian says are important. The
**act-order** heuristic instead visits columns in **descending order of
$\mathrm{diag}(H)$** (a column's own activation "energy"), so the most
important columns are quantized first — while their error-compensation
budget onto the remaining columns is still fully available.

### Algorithm

Given $W \in \mathbb{R}^{d_{out}\times d_{in}}$, $H \in \mathbb{R}^{d_{in}\times d_{in}}$
(symmetric PSD), bit width $b$, and relative dampening $\delta$:

1. **Act-order permutation**: $\pi = \mathrm{argsort}(-\mathrm{diag}(H))$
   (descending; ties broken by smaller original index).
2. **Dampen and invert**: reorder $H$ by $\pi$ into $H_\pi = H[\pi,\pi]$, add
   $\delta \cdot \overline{\mathrm{diag}(H_\pi)}$ to its diagonal, and invert:
   $H_\pi^{-1}$.
3. **Per-column quantization params** (computed once, from the *original*
   $W$, before any GPTQ updates): for column $c$, with
   $m=\min(0,\min W_{:,c})$, $M=\max(0,\max W_{:,c})$,
   $$
   s_c = \frac{M-m}{2^{b}-1}\ (\text{or } 1 \text{ if } M=m), \qquad
   z_c = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{-m}{s_c}\right), 0, 2^{b}-1\right)
   $$
   and $\mathrm{quant}(w) = \left(\mathrm{clip}(\mathrm{round}(w/s_c)+z_c,\,0,\,2^{b}-1) - z_c\right) s_c$.
4. **Sequential update**: let $\tilde W = W[:, \pi]$. For $i = 0,\dots,d_{in}-1$
   (in the permuted, i.e. act-order, index order):
   $$
   q = \mathrm{quant}_{\pi_i}(\tilde W_{:,i}), \qquad
   e = \frac{\tilde W_{:,i} - q}{(H_\pi^{-1})_{ii}}
   $$
   $$
   \tilde W_{:,i} \leftarrow q, \qquad
   \tilde W_{:,i+1:} \leftarrow \tilde W_{:,i+1:} - e \,(H_\pi^{-1})_{i,\,i+1:}
   $$
5. **Un-permute**: $\hat W = \tilde W[:, \pi^{-1}]$.

## Task

Implement `gptq_act_order`:

```python
def gptq_act_order(W: np.ndarray, H: np.ndarray, nbits: int, damp: float) -> tuple[np.ndarray, float]:
    ...
```

* `W` — `(d_out, d_in)` weight matrix.
* `H` — `(d_in, d_in)` symmetric PSD calibration Hessian.
* `nbits` — bits per column (unsigned affine quantization, as above).
* `damp` — relative Hessian dampening $\delta$.

Return `(perm, mse)`:

* `perm` — integer array of length `d_in`, the act-order permutation $\pi$.
* `mse` — `float`, $\frac{1}{d_{out}\,d_{in}}\sum (\hat W - W)^2$ for the
  reconstruction $\hat W$ produced by running the algorithm above in that
  column order.

## Example

```python
import numpy as np
rng = np.random.default_rng(0)
X = rng.normal(size=(20, 4))
H = X.T @ X
W = rng.normal(size=(3, 4))
perm, mse = gptq_act_order(W, H, nbits=4, damp=0.01)
# perm is np.argsort(-np.diag(H)) as an int array of length 4
```

## What the gate checks

* **exact_match** — your `perm` must equal
  `np.argsort(-np.diag(H))` exactly (integer array equality) on every
  random trial.
* **rel_err** — the relative difference between your `mse` and the value
  produced by re-running the full algorithm above (dampen, invert, quantize
  in act-order, error-compensate) with a NumPy oracle, on the same `(W, H,
  nbits, damp)`.
