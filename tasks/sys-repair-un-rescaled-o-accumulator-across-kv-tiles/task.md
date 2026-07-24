## Context

FlashAttention computes scaled dot-product attention without storing the full attention
matrix. The dense operation is

$$
O = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V .
$$

A tiled kernel processes the key and value matrices in blocks. For each query row,
the kernel maintains a running softmax maximum $m$, normalization accumulator $l$,
and output accumulator $O$.

When a new tile produces a larger running maximum $m'$, the previous state must be
rescaled. The correction factor is

$$
\alpha = e^{m-m'} .
$$

The state update is

$$
l' = \alpha l + \sum_j e^{s_j-m'}
$$

and

$$
O' = \alpha O + \sum_j e^{s_j-m'}v_j .
$$

A common implementation bug is to update $l$ with $\alpha$ but forget to rescale
the existing $O$ accumulator. This causes earlier tiles to have an incorrect weight
whenever a later tile contains larger attention scores.

## Task

Repair `flash_attention_tiled(Q, K, V, tile_size)`.

The function receives:
- `Q`: a NumPy array of shape $(n,d)$,
- `K`: a NumPy array of shape $(m,d)$,
- `V`: a NumPy array of shape $(m,d)$,
- `tile_size`: a positive integer.

Return the tiled FlashAttention output with shape $(n,d)$ and dtype `float64`.

The implementation should keep the running tiled softmax structure. Do not replace it
with a direct dense attention expression.

```python
def flash_attention_tiled(Q: np.ndarray, K: np.ndarray, V: np.ndarray, tile_size: int) -> np.ndarray:
    ...
```

## Example

```python
import numpy as np

Q = np.array([[1.0, 0.0], [0.0, 1.0]])
K = np.array([[1.0, 0.0], [0.0, 1.0]])
V = np.array([[2.0, 0.0], [0.0, 3.0]])

O = flash_attention_tiled(Q, K, V, 1)
```

The returned matrix must match

$$
\mathrm{softmax}(QK^\top/\sqrt{d})V .
$$

## What the gate checks

The gate builds a dense NumPy attention reference and compares the repaired tiled
implementation using

$$
\mathrm{rel\_err} =
\frac{\lVert O_{\mathrm{candidate}}-O_{\mathrm{reference}}\rVert}
{\lVert O_{\mathrm{reference}}\rVert + 10^{-12}} .
$$

The fixtures contain KV tiles where later tiles create cross-tile maximum jumps.
The result must satisfy

$$
\mathrm{rel\_err} \le 10^{-5}.
$$

The provided implementation fails because it does not rescale the running output
accumulator when the running maximum changes.
