## Context

**Automatic Mixed Precision (AMP)** training scales the loss by a large factor $s$
(e.g., $s = 2^{15}$) before the backward pass to prevent fp16 gradient underflow.
After accumulation the raw gradients are $\tilde{g} = s \cdot g$ where $g$ is the
true gradient.

Before the optimizer can apply the update rule

$$w \leftarrow w - \eta \, g$$

the gradients must be **unscaled**:

$$g = \frac{\tilde{g}}{s}$$

If the optimizer receives the scaled gradients directly it applies a step that is
$s$ times too large, corrupting the weights catastrophically.

The canonical fix is to divide every gradient by the scale factor before
`optimizer.step()`:

```python
for p in params:
    if p.grad is not None:
        p.grad /= scale
```

## Task

The buggy function below performs a gradient-descent step on **scaled** gradients.
Fix it by unscaling every gradient by `scale` before applying the update.

```python
def optimizer_step(params, grads, scale, lr):
    # BUG: grads are still scaled — divide by scale before applying
    for i, (p, g) in enumerate(zip(params, grads)):
        params[i] = p - lr * g          # <-- g should be g / scale
    return params
```

The function takes:
- `params`: list of float64 NumPy arrays (model weights)
- `grads`: list of float64 NumPy arrays (scaled gradients, i.e. $\tilde{g} = s \cdot g$)
- `scale`: float scalar (the AMP loss scale $s$)
- `lr`: float scalar (learning rate $\eta$)

It should return the updated `params` list with weights updated as
$w \leftarrow w - \eta \cdot (\tilde{g} / s)$.

## Example

```python
import numpy as np
params = [np.array([1.0, 2.0])]
grads  = [np.array([32768.0, -32768.0])]   # scaled by s=32768
scale  = 32768.0
lr     = 0.01
result = optimizer_step(params, grads, scale, lr)
# result[0] ≈ [1.0 - 0.01*1.0, 2.0 - 0.01*(-1.0)] = [0.99, 2.01]
```

## What the gate checks

`check.py` creates random parameters and scaled gradients, computes the reference
update $w - \eta \cdot (\tilde{g}/s)$ with NumPy, and checks that your output
matches within $\mathrm{rel\_err} \le 10^{-5}$.
