## Context

KV-cache and activation quantization usually can't assume the data is
zero-centered (post-ReLU activations, attention scores, etc. are often
one-sided). **Asymmetric affine quantization** handles this with an
explicit integer **zero-point** $z$: the code that represents the real
value $0.0$. Getting $z$ right matters — if the representable float
range doesn't include $0$, the zero-point has to be clamped into the
valid integer range, which silently breaks the round-trip accuracy
guarantee. The standard fix, used throughout production quantizers
(PyTorch's `fake_quantize`, TensorRT, TFLite), is to **always widen the
calibration range to include zero** before computing the scale:
$$
m = \min(0, \min_i x_i), \qquad M = \max(0, \max_i x_i).
$$

### Formulas

$$
s = \frac{M - m}{q_{max} - q_{min}} \quad (\text{use } 1 \text{ if } M=m)
\qquad
z = \mathrm{clip}\big(\mathrm{round}(q_{min} - m/s),\ q_{min},\ q_{max}\big)
$$
$$
q_i = \mathrm{clip}\big(\mathrm{round}(x_i/s + z),\ q_{min},\ q_{max}\big)
\qquad
\widehat x_i = (q_i - z)\cdot s
$$

Because the range always includes $0$, $z$'s natural (unclamped) value
is already inside $[q_{min}, q_{max}]$ — so the clip on $z$ never
actually fires, and every reconstructed value satisfies
$|x_i - \widehat x_i| \le s/2$.

## Task

Implement:

```python
def affine_quant_dequant(x: np.ndarray, qmin: int, qmax: int) -> np.ndarray:
    ...
```

* `x` — 1-D array of values (any sign).
* `qmin`, `qmax` — integer code bounds (e.g. `0, 255` for `uint8`).

Return the dequantized array $\widehat x$ computed via the formulas
above.

## Example

```python
import numpy as np
x = np.array([-2.0, -1.0, 0.5, 3.0])
xhat = affine_quant_dequant(x, qmin=0, qmax=255)
# m=-2.0 (already <=0), M=3.0 (already >=0); s=5/255
# every |x_i - xhat_i| <= s/2
```

## What the gate checks

* **max_abs_err** — your dequantized output must match a NumPy oracle
  implementing the exact formulas above to within $10^{-6}$ absolute
  error, over several random vectors and scales (fixed seed).
* **max_step_ratio** — for every case, $\max_i |x_i - \widehat x_i| /
  s \le 0.5$ (up to a tiny numerical slack) — the fundamental accuracy
  guarantee of round-to-nearest affine quantization, which a formula bug
  (e.g. forgetting to widen the range to include 0, or an off-by-one in
  the zero-point) will typically blow past.
