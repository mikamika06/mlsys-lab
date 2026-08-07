## Context

Knowledge distillation methods often combine a loss on the model output logits with a loss
on an intermediate representation. Let teacher logits be $z_t$, student logits be $z_s$,
teacher hidden states be $h_t$, and student hidden states be $h_s$.

The combined objective is

$$
L = \mathrm{KL}(\mathrm{softmax}(z_t) \,\|\, \mathrm{softmax}(z_s))
+ \beta \frac{1}{N}\sum_i (h_{s,i}-h_{t,i})^2 .
$$

The first term matches the output distributions. The second term encourages the
student's intermediate features to remain close to the teacher's features.

For a vector of logits $z$, the softmax function is

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}} .
$$

The implementation must return the loss and the gradients with respect to the student
logits and student hidden states.

## Task

Implement `combined_logit_intermediate_loss`:

```python
def combined_logit_intermediate_loss(teacher_logits: list[list[float]], student_logits: list[list[float]], teacher_hidden: list[list[float]], student_hidden: list[list[float]], beta: float) -> tuple[float, list[list[float]], list[list[float]]]:
    ...
```

Return a tuple:

```python
(loss, grad_student_logits, grad_student_hidden)
```

where:

- `loss` is the scalar combined loss as a Python float.
- `grad_student_logits` has the same shape as `student_logits`.
- `grad_student_hidden` has the same shape as `student_hidden`.

Use Python operations. The KL term should be computed as

$$
\sum_i p_i(\log(p_i)-\log(q_i))
$$

where $p$ is the teacher softmax distribution and $q$ is the student softmax distribution.

The hidden-state gradient should be the derivative of the mean squared error term:

$$
\frac{\partial L}{\partial h_s} =
\frac{2\beta}{M}(h_s-h_t),
$$

where $M$ is the total number of hidden values.

## Example

```python

teacher_logits = [[2.0, 0.5]]
student_logits = [[1.5, 0.7]]
teacher_hidden = [[1.0, 2.0]]
student_hidden = [[0.5, 2.5]]

loss, dlogits, dhidden = combined_logit_intermediate_loss(
    teacher_logits,
    student_logits,
    teacher_hidden,
    student_hidden,
    0.1,
)
```

The returned gradients describe how the loss changes when the student outputs are
perturbed.

## What the gate checks

The gate computes the same loss with a Python oracle and computes numerical gradients
using central finite differences:

$$
\frac{\partial L}{\partial x_i}
\approx
\frac{L(x_i+\epsilon)-L(x_i-\epsilon)}{2\epsilon}.
$$

The returned loss must match the oracle loss with maximum absolute error below
$10^{-10}$. The returned gradients for student logits and hidden states must match
the finite-difference gradients with maximum absolute error below $10^{-5}$.
