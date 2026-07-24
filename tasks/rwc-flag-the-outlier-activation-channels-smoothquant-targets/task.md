## Context

SmoothQuant is a quantization technique that reduces the dynamic range of activations by scaling each channel separately.  
For a batch of activations $X \in \mathbb{R}^{n\times c}$ (with $n$ samples and $c$ channels) the per‑channel maximum absolute value is

$$m_j = \max_{i=1,\dots,n} |X_{ij}|, \qquad j = 1,\dots,c.$$

If a channel’s $m_j$ is far larger than the typical scale of other channels it can dominate the quantization step and degrade accuracy.  
A simple heuristic to detect such *outlier* channels is to compare each $m_j$ against a multiple of the median of all $m_j$:

$$\text{threshold} = \alpha \cdot \operatorname{median}(m),$$

where $\alpha > 1$ (commonly $\alpha=3.0$).  
Channels with $m_j > \text{threshold}$ are flagged as outliers.

## Task

Implement `flag_outliers`:

```python
def flag_outliers(X: np.ndarray, factor: float = 3.0) -> np.ndarray:
    ...
```

The function receives a 2‑D NumPy array $X$ of shape $(n,c)$ and returns a boolean mask of length $c$.  
A value `True` indicates that the corresponding channel is an outlier according to the rule above.  
Use only vectorised NumPy operations; no explicit Python loops.

## Example

```python
import numpy as np
from flag_outliers import flag_outliers

X = np.array([[0, 1, 10],
              [2, 3, 12],
              [4, 5, 13]])

mask = flag_outliers(X, factor=3.0)
print(mask)          # [False False  True]
```

Here the third channel has $m_3 = 13$ while $\operatorname{median}(m)=2$, so $13 > 3 \times 2$ and it is flagged.

## What the gate checks

The grader computes a reference mask using NumPy’s median and max functions on randomly generated tensors.  
Your implementation must return exactly the same boolean array for all test cases; otherwise the `exact_match` metric fails.
