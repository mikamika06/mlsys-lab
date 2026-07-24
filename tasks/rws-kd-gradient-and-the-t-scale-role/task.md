## Context

Knowledge distillation trains a student model to match a teacher model's output distribution. For a temperature $T$, logits are softened as

$$
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)} .
$$

The distillation loss is the KL divergence between teacher and student distributions with a compensating scale factor:

$$
L = T^2 \sum_i p_i^{(t)} \log\frac{p_i^{(t)}}{p_i^{(s)}} .
$$

Without the $T^2$ factor, the gradient magnitude with respect to student logits decreases approximately as $1/T^2$. Multiplying the loss by $T^2$ keeps the gradient scale comparable across different temperatures.

The gradient of the KL term can be derived analytically, but production implementations often need a reliable numerical equivalent that follows the same loss definition.

## Task

Implement `kd_gradient(student_logits, teacher_logits, labels, T, scale_t2=True)`.

The inputs are NumPy arrays:

- `student_logits` has shape $(n, c)$ and contains student model logits.
- `teacher_logits` has shape $(n, c)$ and contains teacher model logits.
- `labels` has shape $(n,)$ and contains integer class labels. It is provided as part of a training batch contract but is not used by the pure KD loss.
- `T` is a positive temperature.
- `scale_t2` controls whether the $T^2$ multiplier is applied.

Return a NumPy array with the same shape as `student_logits` containing $\frac{\partial L}{\partial s}$, the gradient of the KD loss with respect to the student logits.

Use float64 calculations.

## Example

```python
import numpy as np

student = np.array([[1.0, 0.0, -1.0]])
teacher = np.array([[0.5, 0.2, -0.3]])
labels = np.array([0])

grad = kd_gradient(student, teacher, labels, 2.0)

# grad is the student-logit gradient of:
# 2^2 * KL(softmax(teacher/2) || softmax(student/2))
```

## What the gate checks

The gate computes the reference gradient using a central finite-difference oracle over the KD loss. The returned gradient must have maximum absolute error below $10^{-5}$.

The gate also checks the role of the $T^2$ factor. Gradients produced with `scale_t2=True` at two temperatures should keep a similar magnitude, while removing the factor should reduce gradient magnitude approximately according to the expected $1/T^2$ scaling.
