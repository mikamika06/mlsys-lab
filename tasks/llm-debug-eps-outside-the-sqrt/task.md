## Context

**Layer Normalization** normalizes a vector $x \in \mathbb{R}^d$ to zero mean and unit
variance, then applies a learned affine transformation:

$$\mathrm{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sigma + \varepsilon} + \beta$$

where $\mu = \frac{1}{d}\sum_i x_i$ and $\sigma = \sqrt{\frac{1}{d}\sum_i (x_i - \mu)^2}$.

The $\varepsilon > 0$ (typically $10^{-5}$) prevents division by zero.  The **correct**
placement of $\varepsilon$ is **inside** the square root:

$$\sigma = \sqrt{\frac{1}{d}\sum_i (x_i - \mu)^2 + \varepsilon}$$

A common bug places $\varepsilon$ **outside** the square root:

$$\text{(wrong)} \quad \sigma_{\text{wrong}} = \sqrt{\frac{1}{d}\sum_i (x_i - \mu)^2} + \varepsilon$$

For vectors with small variance this creates a systematic error: the denominator is too
large, so the normalized output is too small.

## Task

The buggy function below uses `sqrt(var) + eps`.  Fix it so that $\varepsilon$ is
**inside** the square root: `sqrt(var + eps)`.

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    mu  = x.mean()
    var = ((x - mu) ** 2).mean()
    # BUG: eps is outside the sqrt
    std = var ** 0.5 + eps          # <-- fix this line
    return gamma * (x - mu) / std + beta
```

Fix the single line so that `std = (var + eps) ** 0.5`.

## Example

```python
import numpy as np
x     = np.array([1.0, 2.0, 3.0])
gamma = np.ones(3)
beta  = np.zeros(3)
# Correct result:
layer_norm(x, gamma, beta)
# array([-1.22474487,  0.        ,  1.22474487])
```

## What the gate checks

`check.py` generates random inputs with small variance (where the bug is most visible),
computes the reference output using `sqrt(var + eps)`, and checks that your output
matches within $\mathrm{max\_abs\_err} \le 10^{-6}$.
