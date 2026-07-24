## Context

Two common int8 inference schemes for a linear layer $y = xW^\top$ trade
off differently:

* **Weight-only** quantization (e.g. `bitsandbytes` LLM.int8()-style
  weight paths, GPTQ/AWQ inference kernels) quantizes only $W$ and
  dequantizes it before the matmul; activations $x$ stay in full
  precision throughout.
* **Dynamic (activation) quantization** (e.g. PyTorch's
  `torch.quantization.quantize_dynamic`) additionally quantizes the
  activations **on the fly**, per batch, right before the matmul —
  trading some accuracy for a matmul that can run entirely in int8.

Because weight-only leaves $x$ untouched, it strictly avoids one whole
source of quantization error, so for the same weight quantizer it always
attains **lower** output MSE than dynamic quantization — the question is
by how much, which is exactly what this task computes.

### The two schemes

Both use **symmetric int8, one scale per row** (no zero-point):
for a matrix $A$ with rows $A_r$,
$$
s_r = \frac{\max_j |A_{r,j}|}{127}\ \ (\text{use } 1 \text{ if } 0), \qquad
\widehat{A}_{r,j} = \mathrm{clip}(\mathrm{round}(A_{r,j}/s_r),\,-127,\,127)\cdot s_r.
$$

Full precision reference: $y_{fp} = x W^\top$.

* **Weight-only**: quantize only $W$ (row = output channel) to get
  $\widehat W$; $y_{wo} = x\,\widehat{W}^\top$.
* **Dynamic**: additionally quantize $x$ (row = sample) to get
  $\widehat x$; $y_{dyn} = \widehat x\,\widehat{W}^\top$.

Report $\mathrm{MSE}(y_{fp}, y_{wo})$ and $\mathrm{MSE}(y_{fp}, y_{dyn})$.

## Task

Implement:

```python
def weight_only_vs_dynamic_mse(x: np.ndarray, W: np.ndarray) -> tuple[float, float]:
    ...
```

* `x` — `(batch, d_in)` activations.
* `W` — `(d_out, d_in)` weight matrix.

Return `(mse_weight_only, mse_dynamic)` as defined above.

## Example

```python
import numpy as np
rng = np.random.default_rng(0)
x = rng.normal(size=(8, 32))
W = rng.normal(size=(16, 32))
mse_wo, mse_dyn = weight_only_vs_dynamic_mse(x, W)
assert mse_wo < mse_dyn   # activation quantization always adds extra error
```

## What the gate checks

* **max_abs_err** — both `mse_weight_only` and `mse_dynamic` must match a
  NumPy oracle implementing the exact scheme above to within $10^{-6}$
  absolute error, on several random `(x, W)` shapes and scales (fixed
  seed).
* **ordering_ok** — on every case, your `mse_weight_only` must be
  strictly less than your `mse_dynamic`.
