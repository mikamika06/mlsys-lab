## Context

In deep learning, gradients can sometimes become very large during training, leading to numerical instability (the "exploding gradient" problem). To mitigate this, we can apply **gradient clipping**.

A common approach is **global-norm gradient clipping**, where all gradient tensors are scaled by the same factor if their combined global L2 norm exceeds a specified threshold.

Given a list of $n$ gradient tensors $G_1, G_2, \dots, G_n$, the global L2 norm is defined as:

$$ \text{global\_norm} = \sqrt{\sum_{i=1}^n \sum_{x \in G_i} x^2} $$

If the global norm exceeds $\text{max\_norm}$, we compute a scaling coefficient:

$$ \text{coef} = \frac{\text{max\_norm}}{\text{global\_norm} + \epsilon} $$

where we use $\epsilon = 10^{-6}$ to avoid division by zero. If the global norm is within the limit, $\text{coef} = 1.0$.

Finally, we update each gradient tensor:

$$ G_i \leftarrow G_i \times \text{coef} $$

## Task

Write a function `clip_global_norm(grads: List[list[float]], max_norm: float) -> List[list[float]]` that takes a list of gradient tensors and a maximum norm, and returns a new list of clipped gradient tensors.

- You should not modify the input arrays in-place; return new arrays.
- Use $\epsilon = 10^{-6}$ in the denominator when calculating the scale coefficient, as shown in the formula.

## Example

```python

grads = [[3.0, 4.0]]
max_norm = 2.0

clipped_grads = clip_global_norm(grads, max_norm)
# global_norm = sqrt(3^2 + 4^2) = 5.0
# coef = 2.0 / (5.0 + 1e-6) = 0.39999992
# clipped_grads[0] will be approximately [1.2, 1.6]
```

## What the gate checks

- `max_abs_err`: the maximum absolute element-wise difference between your output and the reference implementation's output. The error must be $\le 10^{-9}$.
