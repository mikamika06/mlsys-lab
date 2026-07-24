## Context

NormalFloat (NF4) is a 4-bit data type used in modern quantization (e.g., QLoRA). Unlike a symmetric uniform grid (INT4), NF4 uses an *asymmetric* grid of 16 levels distributed according to the quantiles of a standard normal distribution, with one level forced to exactly zero. This gives higher resolution near zero, where most neural network weights concentrate.

Let $q_i$ for $i = 1, \dots, 16$ be the quantiles of $\mathcal{N}(0,1)$:

$$q_i = \Phi^{-1}\left(\frac{i - 0.5}{16}\right)$$

where $\Phi^{-1}$ is the inverse CDF (ppf). Normalize these quantiles to the interval $[-1, 1]$:

$$\hat{q}_i = \frac{q_i}{\max_j |q_j|}$$

Then locate the index $k$ where $\hat{q}_k$ is closest to 0 and force it to exactly $0.0$. The final level $v_i$ is:

$$v_i = \hat{q}_i \quad \text{for } i \neq k, \qquad v_k = 0.0$$

The result is a 16-element array that matches the published NF4 lookup table.

## Task

Implement `nf4_levels() -> np.ndarray`:

```python
def nf4_levels() -> np.ndarray:
    ...
```

No arguments. Return a NumPy array of shape `(16,)` and dtype `float64` containing the 16 NF4 level values as described above. Rely on `scipy.stats.norm.ppf` to compute the normal quantiles. The implementation must contain no hardcoded constants — derive every value algorithmically.

## Example

```python
import numpy as np
from scipy.stats import norm

levels = nf4_levels()
print(levels[:4])
# Expected: roughly [-1.0, -0.6961928, -0.5250731, -0.3949175]
# The exact values depend on the ppf; your output must match the published NF4
# table to 1e-4 absolute tolerance.
```

## What the gate checks

The gate computes `max_abs_err(reference, submission)` where `reference` is computed by an oracle that follows the exact same algorithm. If every element matches to within $1 \times 10^{-4}$, the gate passes with score 1.0; otherwise it fails with score 0.0. The reference is not hardcoded — it is derived programmatically from `scipy.stats.norm.ppf`.
