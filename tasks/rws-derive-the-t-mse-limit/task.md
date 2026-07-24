## Context

Temperature-scaled knowledge distillation compares a teacher logit vector $z_t$ and a student logit vector $z_s$ using

$$
\mathrm{KL}(p_t^T \Vert p_s^T),
$$

where

$$
p_i^T = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}.
$$

For large temperature $T$, the softmax distribution can be expanded around the uniform distribution. The constant component of logits does not affect softmax, so only centered logits matter:

$$
\bar{z} = z - \frac{1}{n}\sum_i z_i.
$$

The second-order limit is

$$
T^2 \mathrm{KL}(p_t^T \Vert p_s^T)
\rightarrow
\frac{1}{2}\lVert \bar{z}_t-\bar{z}_s\rVert^2 .
$$

This relationship explains why high-temperature distillation is connected to mean squared error between centered logits.

## Task

Implement `t_mse_limit(z_teacher, z_student, temperatures)`.

The function receives two one-dimensional NumPy arrays of logits with the same length and an array of positive temperatures. It must return a tuple:

```python
(
    scaled_kl,
    limit
)
```

where:

- `scaled_kl` is a one-dimensional array containing $T^2\mathrm{KL}(p_t^T \Vert p_s^T)$ for each temperature.
- `limit` is the value $\frac{1}{2}\lVert \bar{z}_t-\bar{z}_s\rVert^2$.

Compute softmax values in a numerically stable way. The returned arrays must use `float64`.

## Example

```python
import numpy as np

teacher = np.array([2.0, 0.5, -1.0])
student = np.array([1.5, 0.7, -0.8])
temperatures = np.array([1.0, 10.0, 1000.0])

scaled_kl, limit = t_mse_limit(teacher, student, temperatures)

# scaled_kl approaches limit as temperature grows.
# limit is approximately 0.5 * ||centered_teacher - centered_student||^2
```

## What the gate checks

The gate builds its own NumPy oracle for the temperature-scaled KL computation and for the centered-logit MSE limit.

It checks that the returned `scaled_kl` values match the oracle across a grid of increasing temperatures with relative error at most $10^{-10}$.

It also checks that the returned `limit` matches the oracle value computed from

$$
\frac{1}{2}\lVert
(z_t-\mathrm{mean}(z_t))-(z_s-\mathrm{mean}(z_s))
\rVert^2 .
$$
