## Context

AdamW updates parameters by maintaining first and second moment estimates of the gradient. For a parameter vector $p$ and gradient $g$, the moments are updated as

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
$$

and

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2 .
$$

The bias-corrected estimates are

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad
\hat{v}_t = \frac{v_t}{1-\beta_2^t}.
$$

The AdamW parameter update is

$$
p_t = p_{t-1} - \alpha
\left(
\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
+
\lambda p_{t-1}
\right),
$$

where $\alpha$ is the learning rate and $\lambda$ is weight decay.

Production training systems sometimes offload optimizer work to another device. A typical implementation copies gradients to CPU memory, performs the optimizer update on CPU-resident optimizer state and parameters, then copies the updated parameters back. The numerical result should match a normal single-device AdamW step.

## Task

Implement `offloaded_adamw_step`:

```python
def offloaded_adamw_step(
    param: np.ndarray,
    grad: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    step: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...
```

The function receives NumPy arrays representing one parameter tensor and optimizer state before an AdamW update. Simulate the offloaded workflow by computing the update using CPU-resident NumPy arrays. Return the updated parameter, first moment, and second moment arrays.

The returned values must numerically match the standard AdamW update. Do not modify the input arrays in place.

## Example

```python
import numpy as np

param = np.array([1.0, -2.0])
grad = np.array([0.1, -0.2])
m = np.zeros(2)
v = np.zeros(2)

new_param, new_m, new_v = offloaded_adamw_step(
    param, grad, m, v, 1, 1e-2, 0.9, 0.999, 1e-8, 0.01
)
```

The result is the same update that would be produced by applying one AdamW step on a single device.

## What the gate checks

The gate computes its own NumPy AdamW oracle for several parameter and gradient tensors. It compares the submitted implementation output with the oracle using the maximum absolute error:

$$
\max_i |x_i-y_i|.
$$

The error must be at most $10^{-6}$ across parameters and optimizer state arrays.
