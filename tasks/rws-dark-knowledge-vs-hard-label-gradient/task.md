## Context

Knowledge distillation (KD) trains a smaller student model using a teacher model's probability distribution instead of only the hard class label.

For student logits $z \in \mathbb{R}^C$, the temperature-scaled softmax is

$$
p_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)} .
$$

The KD loss uses the teacher probabilities $q$ as soft targets:

$$
L_{KD} = -T^2 \sum_i q_i \log(p_i).
$$

The gradient with respect to the student logits is

$$
\frac{\partial L_{KD}}{\partial z_i} = T(p_i-q_i).
$$

A hard-label cross entropy loss uses a one-hot vector $y$:

$$
L_{CE} = -\sum_i y_i \log(p_i),
$$

with gradient

$$
\frac{\partial L_{CE}}{\partial z_i} = p_i-y_i.
$$

The difference is that KD preserves information about non-target classes. A teacher may assign meaningful probability mass to several classes, while a hard label only records the winning class.

## Task

Implement `kd_hard_label_grad`:

```python
def kd_hard_label_grad(
    student_logits: np.ndarray,
    teacher_probs: np.ndarray,
    label: int,
    temperature: float
) -> tuple[np.ndarray, np.ndarray]:
    ...
```

Return a tuple `(kd_grad, ce_grad)`.

`student_logits` is a one-dimensional NumPy array of student logits. `teacher_probs` contains the teacher soft target probabilities and has the same length. `label` is the integer class index from the hard training label. `temperature` is a positive scalar.

Compute:

- `kd_grad`: the gradient of the temperature-scaled KD loss with respect to student logits.
- `ce_grad`: the gradient of standard hard-label cross entropy with respect to student logits.

Return both arrays as `float64`.

## Example

```python
import numpy as np

student = np.array([2.0, 1.0, 0.0])
teacher = np.array([0.70, 0.20, 0.10])

kd, ce = kd_hard_label_grad(student, teacher, 0, 2.0)

# kd contains T * (softmax(student / T) - teacher)
# ce contains softmax(student) - [1, 0, 0]
```

## What the gate checks

The gate computes the oracle gradients from the mathematical definition using NumPy operations.

The returned gradients must match the oracle with maximum absolute error $\le 10^{-6}$.

The fixture also uses a teacher distribution with informative non-argmax probabilities. The gate verifies that the KD and hard-label gradients are not identical by checking their cosine similarity is less than $1$.
