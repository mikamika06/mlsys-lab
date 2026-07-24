## Context

The GELU (Gaussian Error Linear Unit) activation is defined as
$$\operatorname{gelu}(x)=x\,\Phi(x)
=\frac{x}{2}\bigl(1+\tanh(\sqrt{\tfrac{2}{\pi}}\,(x+0.044715\,x^{3}))\bigr),$$
where $\Phi$ is the standard normal CDF. In many neural‑network libraries a bias vector is added to each input before applying GELU, e.g.
$$y = \operatorname{gelu}(x + b).$$

When this operation is implemented as two separate NumPy calls – first `x+b` and then `gelu` – an intermediate array of shape $(n,)$ is materialised. A fused implementation performs the addition and the GELU computation in a single pass over the data, avoiding that temporary allocation and reducing memory traffic.

## Task

Implement the function

```python
def fuse_bias_gelu(x: np.ndarray, bias: np.ndarray) -> np.ndarray:
    ...
```

`x` and `bias` are one‑dimensional NumPy arrays of equal length. The function must return a new array containing $\operatorname{gelu}(x + \text{bias})$ computed in a single vectorised pass. Use only NumPy operations; do not create an intermediate array that holds the biased input before applying GELU.

## Example

```python
import numpy as np
from solution_ref import fuse_bias_gelu  # or your implementation

x = np.array([0.0, 1.0, -1.0], dtype=np.float32)
bias = np.array([0.5, -0.5, 0.2], dtype=np.float32)

y = fuse_bias_gelu(x, bias)
print(y)
# [0.37454012 0.3520653  0.1459356 ]
```

## What the gate checks

Two metrics are evaluated:

* **max_abs_err** – the maximum absolute difference between your output and a reference implementation that first adds the bias then applies GELU. The value must be $\leq 10^{-6}$.
* **op_count** – the number of Python line events executed inside `fuse_bias_gelu`. A fully fused solution should stay below $50$; this ensures that only a single pass over the data is performed.

A correct implementation will satisfy both thresholds. The starter deliberately performs two separate passes and therefore fails one or both gates.
