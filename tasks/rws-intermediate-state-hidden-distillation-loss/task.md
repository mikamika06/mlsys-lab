## Context

Intermediate-state distillation trains a smaller student model to reproduce internal representations from a larger teacher model. A common approach projects the student's hidden state into the teacher representation space before measuring the difference.

Let $H_t \in \mathbb{R}^{n \times d_t}$ be teacher hidden states, $H_s \in \mathbb{R}^{n \times d_s}$ be student hidden states, and $P \in \mathbb{R}^{d_s \times d_t}$ be a learned projection matrix. The projected student states are

$$
\hat{H}_s = H_s P .
$$

The hidden distillation loss is the mean squared error between the teacher states and projected student states:

$$
L = \frac{1}{n d_t} \sum_{i=1}^{n}\sum_{j=1}^{d_t}
(H_{t,ij} - \hat{H}_{s,ij})^2 .
$$

Production implementations compute this with array operations rather than explicit Python loops.

## Task

Implement `hidden_distillation_loss(teacher, student, projection)`:

```python
def hidden_distillation_loss(
    teacher: np.ndarray,
    student: np.ndarray,
    projection: np.ndarray
) -> float:
    ...
```

The inputs are:

- `teacher`: a 2-D NumPy array with shape $(n, d_t)$.
- `student`: a 2-D NumPy array with shape $(n, d_s)$.
- `projection`: a 2-D NumPy array with shape $(d_s, d_t)$.

Return the scalar mean squared hidden-state distillation loss between `teacher` and `student @ projection`. Compute the result in `float64`.

## Example

```python
import numpy as np

teacher = np.array([[1.0, 2.0], [0.0, 1.0]])
student = np.array([[1.0], [0.5]])
projection = np.array([[1.0, 2.0]])

loss = hidden_distillation_loss(teacher, student, projection)
# 0.0
```

## What the gate checks

The gate generates several hidden-state cases and compares the returned value against a NumPy oracle that computes the projected student states and mean squared error directly.

The relative error

$$
\mathrm{rel\_err} =
\frac{|x-y|}{|y| + 10^{-12}}
$$

between the learner result and the oracle result must be below $10^{-6}$.
