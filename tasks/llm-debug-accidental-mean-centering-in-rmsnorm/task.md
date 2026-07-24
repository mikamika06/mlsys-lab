## Context

RMSNorm is a lightweight normalization that rescales each row of an input tensor by its root‑mean‑square (RMS) magnitude.  
For a batch matrix $X \in \mathbb{R}^{B\times D}$ the correct operation is

$$
Y_{b} = \frac{X_{b}}{\sqrt{\frac{1}{D}\sum_{d=1}^{D} X_{bd}^{2} + \varepsilon}},
$$

where $\varepsilon>0$ prevents division by zero.  
Unlike LayerNorm, RMSNorm does **not** subtract the mean of each row before scaling; it only uses the squared values.

A common mistake is to copy a LayerNorm implementation and forget that the centering step should be omitted. The resulting function will produce biased outputs.

## Task

Implement `rms_norm`:

```python
def rms_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    ...
```

* `x` is a 2‑D NumPy array of shape $(B,D)$.
* The function must return a new array of the same shape and dtype `float64`.
* No mean subtraction should occur; only divide by the RMS of each row.

## Example

```python
import numpy as np
from rms_norm import rms_norm

X = np.array([[1, 2], [3, 4]], dtype=np.float64)
Y = rms_norm(X)
print(Y)
# [[0.4472136 0.89442719]
#  [0.9486833 1.8973666]]
```

## What the gate checks

The grader computes a reference implementation using NumPy and compares your output with it.  
It reports the maximum absolute difference:

$$
\max_{i,j} |\, \text{your}(i,j) - \text{reference}(i,j)\,|.
$$

Your solution must achieve `max_abs_err <= 1e-6`. Any larger error will fail the gate.
