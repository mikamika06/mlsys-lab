## Context

AdamW maintains first and second moment estimates for gradients. For parameters
$\theta$, gradient $g$, and step index $t$, the moments are updated as

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
$$

and

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2 .
$$

The AdamW parameter update is

$$
\theta_t =
\theta_{t-1}
-
\alpha
\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
-
\alpha\lambda\theta_{t-1},
$$

where $\alpha$ is the learning rate, $\lambda$ is weight decay, and the bias
corrected moments are

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t},
\qquad
\hat{v}_t = \frac{v_t}{1-\beta_2^t}.
$$

Large optimizers reduce memory by storing moment tensors in blockwise 8-bit
form. A tensor block is represented by integer codes and one scale value:

$$
x_i \approx s \, q_i,
$$

where $q_i$ is an int8 value in the range $[-127,127]$ and

$$
s = \frac{\max_i |x_i|}{127}.
$$

Before an optimizer step, the codes are dequantized, the AdamW update is
performed in float32, and the new moments are quantized again.

## Task

Implement `adamw_8bit_step`:

```python
def adamw_8bit_step(
    params,
    grads,
    state,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    block_size,
):
    ...
```

The inputs are 1-D NumPy arrays. `state` is either `None` for initialization or
the dictionary returned by the previous call.

Return `(new_params, new_state)`.

The state dictionary must contain blockwise int8 moment storage:

- `m_codes` and `v_codes`: integer arrays containing quantized moments.
- `m_scales` and `v_scales`: float arrays containing one scale per block.
- `step`: the current optimizer step number.

For each step, dequantize the stored moments, update AdamW moments, update the
parameters, then requantize the moments using blocks of size `block_size`.

## Example

```python
import numpy as np

p = np.array([1.0, -2.0, 3.0], dtype=np.float32)
g = np.array([0.1, -0.2, 0.3], dtype=np.float32)

p, state = adamw_8bit_step(
    p, g, None,
    0.01, 0.9, 0.999, 1e-8, 0.01, 2
)
```

After the first call, `state["m_codes"]` and `state["v_codes"]` store the
compressed optimizer moments rather than full precision arrays.

## What the gate checks

The gate runs several optimizer steps and compares the returned parameter
trajectory against a NumPy oracle implementing blockwise int8 moment
dequantization, AdamW updates, and requantization.

The relative error

$$
\mathrm{rel\_err} =
\frac{\lVert p_{\mathrm{candidate}}-p_{\mathrm{oracle}}\rVert}
{\lVert p_{\mathrm{oracle}}\rVert + 10^{-12}}
$$

must satisfy $\mathrm{rel\_err} \le 10^{-6}$.

Solutions that keep full precision moments, use a single tensor-wide scale, or
skip requantization will not match the oracle trajectory.
