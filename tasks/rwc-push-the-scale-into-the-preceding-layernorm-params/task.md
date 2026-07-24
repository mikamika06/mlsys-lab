## Context

In quantised-inference pipelines a per-feature scale $s \in \mathbb{R}^{d_{\text{in}}}$ is
often introduced between a LayerNorm and the following Linear layer. Rather
than keep the explicit scale node, we can absorb it into the surrounding
parameters. Consider the original (scale-free) pipeline:

$$x \;\xrightarrow{\;\text{LN}\;}\; \gamma \odot \hat{x} + \beta
  \;\xrightarrow{\;\text{Linear}\;}\; W\bigl(\gamma \odot \hat{x} + \beta\bigr) + b,$$

where $\hat{x}_i = (x_i - \mu)/\sigma$ is the LayerNorm normalised input,
$\gamma,\beta \in \mathbb{R}^{d_{\text{in}}}$ are the gain and bias,
$W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ and $b \in \mathbb{R}^{d_{\text{out}}}$
are the Linear parameters, and $\odot$ is element-wise product.

We want to divide $\gamma$ and $\beta$ by $s$ and simultaneously multiply each
column of $W$ by $s$ so the overall output is unchanged. Write
$\text{diag}(s)$ for the diagonal matrix with $s$ on its diagonal. The fused
identity is:

$$W\,\text{diag}(s)\,\bigl(\gamma' \odot \hat{x} + \beta'\bigr)
  = W\,\text{diag}(s)\,\bigl(\tfrac{\gamma}{s} \odot \hat{x} + \tfrac{\beta}{s}\bigr)
  = W\,\bigl(\gamma \odot \hat{x} + \beta\bigr),$$

because element $j$ of the product $\text{diag}(s)(\gamma/s \odot \hat{x} + \beta/s)$ is

$$s_j \cdot \frac{\gamma_j}{s_j}\,\hat{x}_j + s_j \cdot \frac{\beta_j}{s_j}
  = \gamma_j \hat{x}_j + \beta_j.$$

Therefore $W' = W\,\text{diag}(s)$ (column $j$ of $W$ is multiplied by $s_j$),
$\gamma' = \gamma \oslash s$, $\beta' = \beta \oslash s$, and the pipeline output
is preserved exactly (up to floating-point rounding).

## Task

Implement the function

```python
def fuse_scale_into_layernorm(ln_weight, ln_bias, ln_eps, scale, linear_weight, linear_bias):
    ...
```

**Arguments**

| Parameter | Type | Shape | Description |
|---|---|---|---|
| `ln_weight` | `np.ndarray` | `(d_in,)` | LayerNorm $\gamma$ |
| `ln_bias` | `np.ndarray` | `(d_in,)` | LayerNorm $\beta$ |
| `ln_eps` | `float` | scalar | LayerNorm epsilon (unused in param math, but part of the block signature) |
| `scale` | `np.ndarray` | `(d_in,)` | Per-feature scale $s$ |
| `linear_weight` | `np.ndarray` | `(d_out, d_in)` | Linear weight $W$ |
| `linear_bias` | `np.ndarray` | `(d_out,)` | Linear bias $b$ (not folded; not returned) |

**Returns** `(new_ln_weight, new_ln_bias, new_linear_weight)`.

1. Divide $\gamma$ and $\beta$ element-wise by $s$: $\gamma'_j = \gamma_j / s_j$.
2. Multiply each column $j$ of $W$ by $s_j$: $W'_{ij} = W_{ij} \cdot s_j$.
3. Do **not** modify any input array in place.

## Example

```python
import numpy as np
ln_w  = np.array([1.0, 2.0])
ln_b  = np.array([0.5, -0.5])
scale = np.array([0.5, 2.0])
W     = np.ones((2, 2))
b     = np.zeros(2)

new_gw, new_gb, new_W = fuse_scale_into_layernorm(ln_w, ln_b, 1e-5, scale, W, b)
# new_gw ≈ [2.0, 1.0]         (gamma / s)
# new_gb ≈ [1.0, -0.25]       (beta  / s)
# new_W  ≈ [[0.5, 2.0],       (each column of W multiplied by s)
#            [0.5, 2.0]]
```

## What the gate checks

Two numeric gates verified against a NumPy oracle over several random
fixtures (including negative scales and near-zero gamma):

**`ln_param_err`** — global relative $L_2$ error of the student's
$\gamma'$ and $\beta'$ against $\gamma / s$ and $\beta / s$.
Must be $\le 10^{-6}$.

**`output_err`** — global relative $L_2$ error between the unfused
reference output $\text{LN}(x, \gamma, \beta)\;W^{\!\top} + b$ and the
student's fused output $\text{LN}(x, \gamma', \beta')\;W'^{\!\top} + b$
evaluated on random inputs. Must be $\le 10^{-6}$.

If the student forgets to scale $W$ the param gate may still pass, but
the output gate catches the missing factor. If the student multiplies LN
params by $s$ instead of dividing, both gates fail.

Grading helpers available (arena/scorers.py):
"""Deterministic, hardware-independent scorers.

Every scorer takes plain arrays and returns a float. No timing here — speed
metrics live elsewhere and are always same-machine ratios.
"""
from __future__ import annotations

import numpy as np

def _softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)

def mean_kl(ref_logits: np.ndarray, cand_logits: np.ndarray, eps: float = 1e-12) -> float:
    p = _softmax(ref_logits)
    q = _softmax(cand_logits)
    kl = np.sum(p * (np.log(p + eps) - np.log(q + eps)), axis=-1)
    return float(np.mean(kl))

def argmax_agreement(ref_logits: np.ndarray, cand_logits: np.ndarray) -> float:
    a = np.argmax(np.asarray(ref_logits), axis=-1)
    b = np.argmax(np.asarray(cand_logits), axis=-1)
    return float(np.mean(a == b))

def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean((a - b) ** 2))

def size_ratio(original: np.ndarray, *quantized_parts: np.ndarray) -> float:
    orig_bytes = np.asarray(original).nbytes
    q_bytes = sum(int(np.asarray(p).nbytes) for p in quantized_parts)
    if q_bytes == 0:
        return float("inf")
    return float(orig_bytes / q_bytes)

def rel_err(original: np.ndarray, approx: np.ndarray) -> float:
    a = np.asarray(original, dtype=np.float64).ravel()
    b = np.asarray(approx, dtype=np.float64).ravel()
    return float(np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-12))

def channel_rel_err(W: np.ndarray, W_hat: np.ndarray, axis: int = 1) -> float:
    W = np.asarray(W, dtype=np.float64)
    W_hat = np.asarray(W_hat, dtype=np.float64)
    num = np.linalg.norm(W_hat - W, axis=axis)
    den = np.linalg.norm(W, axis=axis) + 1e-12
    return float(np.mean(num / den))

def max_abs_err(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)))

def byte_exact_fraction(a: bytes | np.ndarray, b: bytes | np.ndarray) -> float:
    ba = a.tobytes() if isinstance(a, np.ndarray) else bytes(a)
    bb = b.tobytes() if isinstance(b, np.ndarray) else bytes(b)
    if len(ba) != len(bb) or len(ba) == 0:
        return 0.0
    same = sum(1 for x, y in zip(ba, bb) if x == y)
    return same / len(ba)
