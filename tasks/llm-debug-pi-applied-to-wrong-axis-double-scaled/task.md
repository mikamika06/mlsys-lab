## Context

**Rotary Position Embedding (RoPE)** encodes position information by rotating pairs of
hidden dimensions.  For a query/key vector $x \in \mathbb{R}^d$ at position $p$, the
$k$-th pair $(x_{2k}, x_{2k+1})$ is rotated by angle $p \cdot \theta_k$ where

$$\theta_k = \frac{1}{10000^{2k/d}}$$

**Position Interpolation (PI)** extends the context window from the training length
$L_{\text{train}}$ to a new length $L_{\text{new}}$ by scaling positions before
computing RoPE angles:

$$p' = p \cdot \frac{L_{\text{train}}}{L_{\text{new}}}$$

Two common bugs arise:
1. **Double-scaling**: applying the scale factor to the *frequencies* $\theta_k$ **and**
   again to the position $p$ (so the net effect is $p^2 / L$).
2. **Wrong axis**: scaling over the dimension axis instead of the position axis, which
   scrambles the frequency distribution.

The correct approach is to scale positions only:

$$\phi_{p,k} = \left(p \cdot \frac{L_{\text{train}}}{L_{\text{new}}}\right) \cdot \theta_k$$

## Task

The buggy function below double-scales by multiplying both the position and the
frequency:

```python
def rope_pi(seq_len, dim, L_train, L_new):
    # positions
    pos = np.arange(seq_len, dtype=np.float64)
    scale = L_train / L_new
    # BUG: scale applied to BOTH pos AND theta
    pos_scaled = pos * scale
    k = np.arange(dim // 2, dtype=np.float64)
    theta = 1.0 / (10000.0 ** (2 * k / dim)) * scale   # <-- extra * scale is the bug
    angles = np.outer(pos_scaled, theta)
    cos = np.cos(angles)
    sin = np.sin(angles)
    return cos, sin
```

Fix it so that the scale appears **only** in the position:

```python
pos_scaled = pos * scale
theta = 1.0 / (10000.0 ** (2 * k / dim))   # no scale here
```

## Example

```python
import numpy as np
cos, sin = rope_pi(4, 8, L_train=2048, L_new=4096)
# cos.shape == (4, 4), sin.shape == (4, 4)
# At position 0: all angles are 0, so cos row 0 == [1, 1, 1, 1]
```

## What the gate checks

`check.py` computes the reference `(cos, sin)` tensors using the correct formula
(scale only the positions), then checks that your output matches within
$\mathrm{max\_abs\_err} \le 10^{-6}$.
