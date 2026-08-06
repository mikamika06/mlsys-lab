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
import math

def rope_pi(seq_len: int, dim: int, L_train: float, L_new: float) -> tuple[list[list[float]], list[list[float]]]:
    # positions
    pos = list(range(seq_len))
    scale = L_train / L_new
    # BUG: scale applied to BOTH pos AND theta
    pos_scaled = [p * scale for p in pos]
    k = list(range(dim // 2))
    theta = [1.0 / (10000.0 ** (2 * i / dim)) * scale for i in k]   # <-- extra * scale is the bug
    angles = [[p * t for t in theta] for p in pos_scaled]
    cos = [[math.cos(a) for a in row] for row in angles]
    sin = [[math.sin(a) for a in row] for row in angles]
    return cos, sin
```

Fix it so that the scale appears **only** in the position:

```python
pos_scaled = [p * scale for p in pos]
theta = [1.0 / (10000.0 ** (2 * i / dim)) for i in k]   # no scale here
```

## Example

```python
cos, sin = rope_pi(4, 8, L_train=2048, L_new=4096)
# len(cos) == 4, len(cos[0]) == 4
# At position 0: all angles are 0, so cos row 0 == [1, 1, 1, 1]
```

## What the gate checks

`check.py` computes the reference `(cos, sin)` tensors using the correct formula (scale only the positions), then checks that your output matches within $\mathrm{max\_abs\_err} \le 10^{-6}$.
