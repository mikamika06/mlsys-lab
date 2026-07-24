## Context

Mixed-precision training often stores model weights in two forms. The forward and backward passes may use lower precision values such as `float16`, while an `float32` master copy is updated to preserve numerical accuracy.

A gradient update with learning rate $\eta$ is

$$
w_{\mathrm{new}} = w - \eta g .
$$

When gradients are produced in `float16`, the update should first use a `float32` representation of the gradient:

$$
w_{\mathrm{master,new}} =
\mathrm{float32}(w_{\mathrm{master}})
-
\eta \cdot \mathrm{float32}(g_{\mathrm{fp16}}).
$$

The value used by the next low-precision forward pass is then the re-cast weight

$$
w_{\mathrm{model,new}} = \mathrm{float16}(w_{\mathrm{master,new}}).
$$

Keeping the update in the master precision avoids accumulating rounding error from repeated low-precision updates.

## Task

Implement `mixed_precision_step(master_weight, grad_fp16, lr)`:

```python
def mixed_precision_step(master_weight: np.ndarray,
                         grad_fp16: np.ndarray,
                         lr: float) -> tuple[np.ndarray, np.ndarray]:
    ...
```

The inputs are:

- `master_weight`: a `float32` NumPy array containing the master parameters.
- `grad_fp16`: a `float16` NumPy array containing the computed gradients.
- `lr`: a Python float learning rate.

Return a pair:

1. The updated `float32` master weights.
2. The updated `float16` model weights obtained by casting the updated master weights.

The computation must perform the subtraction in `float32`, not by updating a `float16` copy.

## Example

```python
import numpy as np

w = np.array([1.0, -2.0], dtype=np.float32)
g = np.array([0.5, 0.25], dtype=np.float16)

master, model = mixed_precision_step(w, g, 0.1)

# master:
# [0.95, -2.025] as float32
#
# model:
# [0.95, -2.025] rounded to float16
```

## What the gate checks

The gate computes the expected mixed-precision update using NumPy operations in `float32`. The returned master weights and model weights are compared with the oracle result using maximum absolute error.

The reported metric $\mathrm{max\_abs\_err}$ must satisfy

$$
\mathrm{max\_abs\_err} < 10^{-6}.
$$

Solutions that update through `float16` intermediate values lose precision and do not pass.
