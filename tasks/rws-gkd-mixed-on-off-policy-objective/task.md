## Context

Generalized Knowledge Distillation (GKD) can combine training signals from two sources:
student rollouts and reference data. For each token, the student logits define a
distribution over a vocabulary of size $V$. Given logits $z \in \mathbb{R}^{V}$,
the cross-entropy loss for target token $y$ is

$$
\mathrm{CE}(z, y) = -\log \frac{\exp(z_y)}{\sum_{j=1}^{V}\exp(z_j)}.
$$

A mixed objective uses an interpolation coefficient $\lambda$ to combine the
on-policy rollout target and the off-policy data target:

$$
L_i =
\lambda \,\mathrm{CE}(z_i, y_i^{on}) +
(1-\lambda)\,\mathrm{CE}(z_i, y_i^{off}).
$$

The final loss is the mean over all tokens:

$$
L = \frac{1}{T}\sum_{i=1}^{T} L_i.
$$

This objective lets training balance generated student trajectories and fixed
dataset supervision.

## Task

Implement `gkd_mixed_loss`:

```python
def gkd_mixed_loss(
    student_logits: np.ndarray,
    on_policy_targets: np.ndarray,
    off_policy_targets: np.ndarray,
    lam: float,
) -> float:
    ...
```

`student_logits` has shape $(T, V)$ and contains the student model logits for
each token. Both target arrays have shape $(T,)$ and contain integer vocabulary
indices. `lam` is a float in $[0,1]$.

Return the scalar mean mixed objective. Compute the loss in `float64` and use a
numerically stable log-softmax calculation.

## Example

```python
import numpy as np

logits = np.array([[2.0, 0.0], [0.0, 2.0]])
on_targets = np.array([0, 1])
off_targets = np.array([1, 0])

loss = gkd_mixed_loss(logits, on_targets, off_targets, 0.5)
```

The returned value is the average of the two-token mixed cross-entropy objective.

## What the gate checks

The gate creates several token batches with on-policy targets, off-policy
targets, a mixing coefficient $\lambda$, and student logits. It recomputes the
objective with an independent NumPy oracle and compares the returned scalar
using relative error. A correct implementation must satisfy
$\mathrm{rel\_err} \le 10^{-6}$.
