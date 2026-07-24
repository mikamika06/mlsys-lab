## Context

Quantization reduces the precision of neural network weights to save memory and accelerate inference.  
A common target is **int4**, a signed 4‑bit representation that can encode values in the range $[-8,7]$.  
Two popular quantization schemes for int4 are:

* **RTN** (Round‑to‑Nearest): each weight $w$ is scaled by a global factor $s$, rounded to the nearest integer, clipped to $[-8,7]$, and then dequantized as $\hat w = q\,s$.

* **AWQ** (Activation‑Aware Weight Quantization): uses the same initial scaling but refines it per channel to minimise mean‑square error.  
  For a channel vector $x$, let $q=\operatorname{clip}\!\bigl(\operatorname{round}(x/s_0),-8,7\bigr)$ with $s_0=\max|x|/7$.  
  The optimal scale is
  $$ s_{\text{opt}} = \frac{\sum_i x_i\,q_i}{\sum_i q_i^2}\,. $$
  Dequantisation then uses $\hat x = q\,s_{\text{opt}}$.

The quality of a quantization method can be measured by the **relative error**
$$
\mathrm{rel\_err}(x,\hat x) \;=\;
\frac{\lVert x-\hat x\rVert_2}{\lVert x\rVert_2 + 10^{-12}}\;,
$$
averaged over all channels.

## Task

Implement the function

```python
def awq_vs_rtn_quality(w: np.ndarray) -> tuple[float, float]:
    ...
```

`w` is a 2‑D NumPy array of shape $(C,N)$ containing floating‑point weights.  
Return a tuple `(rel_err_awq, rel_err_rtn)` where each entry is the mean relative error over all channels for AWQ and RTN respectively.

The implementation must use only vectorised NumPy operations; no explicit Python loops over elements are allowed.

## Example

```python
import numpy as np
w = np.array([[0.1, -0.3], [2.5, 1.7]])
rel_err_awq, rel_err_rtn = awq_vs_rtn_quality(w)
print(rel_err_awq, rel_err_rtn)   # e.g., (0.0123456, 0.0234567)
```

## What the gate checks

Two gates are applied:

1. **AWQ relative error** – the value returned for AWQ must match the reference within a tolerance of $10^{-9}$.

2. **RTN relative error** – the value returned for RTN must also match the reference within $10^{-9}$.

The reference is computed by the grader using the exact algorithm described above; no hard‑coded numbers are used.  A correct implementation will therefore pass both gates, whereas any deviation (e.g., missing the optimal scaling step in AWQ) will cause a gate failure.
