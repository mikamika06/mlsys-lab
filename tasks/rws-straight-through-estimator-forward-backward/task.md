## Context

Many compression techniques need a genuinely *hard*, non-differentiable
decision in the forward pass — snapping to one codebook entry, routing to
one expert, picking one bit pattern — but still need a usable gradient to
flow backward through that decision during training. The straight-through
estimator (STE) resolves this by using two different functions for the two
passes.

Forward, it emits the hard one-hot vector of the arg-max:

$$
y^{\text{hard}}_i = \begin{cases} 1 & i = \arg\max_j z_j \\ 0 & \text{otherwise} \end{cases}
$$

Backward, it pretends the forward pass had instead been the *soft*
$\operatorname{softmax}(z)$, and back-propagates the upstream gradient $v$
through that softmax's Jacobian $J_{ij} = s_i(\delta_{ij} - s_j)$, where
$s = \operatorname{softmax}(z)$:

$$
\frac{\partial \mathcal{L}}{\partial z} = J^{\top} v, \qquad\text{which simplifies to}\qquad
\frac{\partial \mathcal{L}}{\partial z} = s \odot \left(v - \textstyle\sum_k v_k s_k\right).
$$

This lets the network train as if the hard choice were a smooth softmax,
while still using the discrete choice at inference time.

## Task

Implement `ste_argmax(logits, upstream_grad)`.

- `logits`: array of shape `(..., C)`.
- `upstream_grad`: array of shape `(..., C)`, the gradient of the
  downstream loss with respect to the STE's forward output.

Compute:

1. `y_hard`: one-hot vector of `argmax(logits, axis=-1)` per row (ties
   broken by the lowest index — i.e. behave exactly like `np.argmax`).
2. `s = softmax(logits, axis=-1)` (numerically stable: subtract the
   per-row max before exponentiating).
3. `grad_logits = s * (upstream_grad - sum(upstream_grad * s, axis=-1, keepdims=True))`.

Return `(y_hard, grad_logits)`, both `float64` arrays with the same shape
as `logits`.

## Example

```python
import numpy as np

logits = np.array([[1.0, 3.0, 0.5]])
upstream_grad = np.array([[0.1, -0.2, 0.05]])

y_hard, grad_logits = ste_argmax(logits, upstream_grad)
# y_hard == [[0.0, 1.0, 0.0]]  (index 1 has the largest logit)
# grad_logits is NOT one-hot -- it's the softmax-Jacobian-projected
# upstream gradient, spread across all three logits.
```

## What the gate checks

The gate builds two independent oracles across several `(logits,
upstream_grad)` cases, including a row with a tied maximum and an
all-equal row:

- `forward_exact`: your `y_hard` must exactly equal the oracle's one-hot
  argmax on every case (integer equality, no tolerance).
- `grad_rel_err`: your `grad_logits` is compared against a **central
  finite-difference** oracle that perturbs each logit dimension
  independently (`eps = 1e-4`) and measures the resulting change in
  `softmax(logits)`, dotted with `upstream_grad` — an oracle that never
  assumes your (or any) closed-form Jacobian formula is correct. The
  relative L2 error must be at most `1e-5`.

A solution that back-propagates through the *hard* one-hot output (e.g.
returning `grad_logits = upstream_grad` wherever `y_hard` is 1 and 0
elsewhere) passes `forward_exact` but fails `grad_rel_err`, since that is
exactly the "straight-through" shortcut this task is testing you *don't*
take literally.
