## Context

Knowledge distillation trains a student model to match a teacher model. For a
batch of teacher logits $z_t$ and student logits $z_s$, the softened
probabilities use a temperature $T$:

$$
p = \operatorname{softmax}(z_t / T), \qquad q = \operatorname{softmax}(z_s / T).
$$

The distillation loss used by DistilBERT-style training includes a $T^2$
scaling factor:

$$
L = T^2 \left(-\frac{1}{N}\sum_i \sum_j p_{ij}\log(q_{ij})\right).
$$

The gradient with respect to the student logits simplifies to:

$$
\frac{\partial L}{\partial z_s} = \frac{T}{N}(q-p),
$$

where $N$ is the batch size. A common implementation bug is to omit the
$T^2$ factor or to multiply logits by $T$ before softmax instead of dividing.

## Task

Implement `kd_loss_and_grad(teacher_logits, student_logits, T)`:

```python
def kd_loss_and_grad(
    teacher_logits: np.ndarray,
    student_logits: np.ndarray,
    T: float
) -> tuple[float, np.ndarray]:
    ...
```

The inputs are two arrays of shape $(N, C)$ containing teacher and student
logits. Return the scalar distillation loss and the gradient of that loss with
respect to `student_logits`.

Use NumPy operations only. The returned gradient must have the same shape as
`student_logits` and use `float64` values.

## Example

```python
import numpy as np

teacher = np.array([[2.0, 0.0, -1.0]])
student = np.array([[1.0, 0.5, -0.5]])

loss, grad = kd_loss_and_grad(teacher, student, 2.0)
```

The result should represent the temperature-scaled KD objective, not the
ordinary cross-entropy between the raw logits.

## What the gate checks

The grader computes an independent NumPy oracle for the KD loss and analytical
gradient. The returned loss and gradient are compared together using relative
error $\mathrm{rel\_err}$, which must satisfy

$$
\mathrm{rel\_err} < 10^{-6}.
$$

Implementations that omit the $T^2$ multiplier or use
$\operatorname{softmax}(zT)$ instead of $\operatorname{softmax}(z/T)$ will fail.
