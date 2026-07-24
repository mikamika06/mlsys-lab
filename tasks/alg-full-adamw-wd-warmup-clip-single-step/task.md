## Context

AdamW combines Adam's adaptive moment estimates with decoupled weight decay. For a
parameter vector $\theta$, gradient $g_t$, first moment $m_t$, and second moment
$v_t$, the moments are updated as

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2 .
$$

Bias correction gives

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad
\hat{v}_t = \frac{v_t}{1-\beta_2^t}.
$$

The AdamW parameter update is

$$
\theta_t =
\theta_{t-1}
- \alpha_t
\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
- \alpha_t \lambda \theta_{t-1},
$$

where $\lambda$ is the decoupled weight decay coefficient.

A warmup schedule scales the base learning rate during the first steps:

$$
\alpha_t =
\alpha
\min(1, \frac{t}{T_\mathrm{warmup}}).
$$

Before the optimizer update, global norm clipping limits the gradient magnitude:

$$
g_t \leftarrow g_t \frac{c}{\lVert g_t\rVert}
\quad \text{if } \lVert g_t\rVert > c .
$$

## Task

Implement `adamw_single_step(theta, grad, m, v, step, lr, beta1, beta2, eps, weight_decay, warmup_steps, clip_norm)`.

The function receives NumPy arrays containing the current parameters, gradient, and
Adam state. It must return a tuple:

```python
(new_theta, new_m, new_v)
```

All returned arrays must be `float64`.

The implementation should perform exactly one fused optimizer step in this order:

1. Compute the global gradient norm and apply clipping when needed.
2. Update first and second moments.
3. Apply bias correction using the provided integer `step`.
4. Compute the warmup-adjusted learning rate.
5. Apply AdamW's adaptive update and decoupled weight decay.

Do not mutate the input arrays.

## Example

```python
import numpy as np

theta = np.array([1.0, -2.0])
grad = np.array([0.5, -0.5])
m = np.zeros(2)
v = np.zeros(2)

new_theta, new_m, new_v = adamw_single_step(
    theta, grad, m, v,
    step=1,
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    weight_decay=0.01,
    warmup_steps=10,
    clip_norm=1.0,
)
```

## What the gate checks

The gate computes the expected update using an independent NumPy reference
implementation of the AdamW equations. The returned parameter, first moment, and
second moment arrays are compared against the oracle output.

The reported metric is

$$
\max_i |x_i-y_i|
$$

over all returned values. The value must be at most $10^{-8}$.
