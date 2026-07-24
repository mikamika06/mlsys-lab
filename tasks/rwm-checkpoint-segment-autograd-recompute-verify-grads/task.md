## Context

Training deep neural networks often uses activation checkpointing to reduce memory use.
Instead of storing every intermediate activation during the forward pass, a checkpointed
segment stores only the inputs needed to replay the segment later.

For a function $f$ with parameters $\theta$, the normal backward pass uses cached
intermediate values. A checkpointed backward pass discards those values and recomputes

$$
y = f(x, \theta)
$$

during backward before applying the chain rule. The gradients must match the gradients
from ordinary autograd:

$$
\nabla_x L_{\mathrm{checkpoint}} = \nabla_x L_{\mathrm{normal}},
\qquad
\nabla_\theta L_{\mathrm{checkpoint}} = \nabla_\theta L_{\mathrm{normal}} .
$$

This task uses a small two-layer block:

$$
h = \mathrm{ReLU}(xW_1^T + b_1),
$$

$$
y = hW_2^T + b_2 .
$$

The checkpoint implementation should avoid saving the hidden activation $h$ during the
forward pass. During backward it should recompute the segment from saved inputs and
then let autograd calculate gradients.

## Task

Implement `checkpoint_segment(x, w1, b1, w2, b2)`:

```python
def checkpoint_segment(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor], int]:
    ...
```

The function must:

1. Run the two-layer block with a custom `torch.autograd.Function`.
2. Save only the tensors required to recompute the segment.
3. Recompute the forward segment during backward instead of storing the intermediate
   hidden activation.
4. Return the scalar loss `output.sum()`, a list containing gradients for
   `[x, w1, b1, w2, b2]`, and the number of tensors stored by the checkpoint function.

The returned gradients must be detached tensors. The implementation must work with
`torch.float64` tensors.

## Example

```python
import torch

x = torch.tensor([[1.0, -2.0]], dtype=torch.float64, requires_grad=True)
w1 = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
b1 = torch.randn(3, dtype=torch.float64, requires_grad=True)
w2 = torch.randn(1, 3, dtype=torch.float64, requires_grad=True)
b2 = torch.randn(1, dtype=torch.float64, requires_grad=True)

loss, grads, saved = checkpoint_segment(x, w1, b1, w2, b2)
# saved is 5 because only the five recomputation inputs are stored.
# grads contains gradients for x, w1, b1, w2, and b2.
```

## What the gate checks

The grader creates the same MLP block and computes reference gradients using ordinary
PyTorch autograd. It compares the gradients returned by the checkpoint implementation
with the oracle gradients using maximum absolute error.

The grader also computes the expected saved tensor count from the recomputation
requirements of the segment. The checkpoint must store exactly the tensors needed to
replay the block and no intermediate activation tensor.
