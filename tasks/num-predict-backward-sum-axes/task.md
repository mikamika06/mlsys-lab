## Context

Given two vectors $a$ and $b$ of different lengths, broadcasting allows us to perform operations between them. When reducing the result of such an operation along certain axes, the choice of axes to sum can significantly impact the outcome. In the context of automatic differentiation and gradient computation, predicting the correct axes to sum is crucial for accurate gradient propagation.

## Task

Implement `predict_backward_sum_axes(a_shape, b_shape)`:

```python
import numpy as np

def predict_backward_sum_axes(a_shape, b_shape):
    ...
```

It takes two tuples representing the shapes of two NumPy arrays `a` and `b`. The function should return a tuple of two tuples. The first tuple contains the axes to sum for the backward pass of `a`, and the second tuple contains the axes to sum for the backward pass of `b`. The axes should be given in the order they appear in the input shapes.

## Example

```python
a_shape = (3, 4)
b_shape = (1, 4)
print(predict_backward_sum_axes(a_shape, b_shape))
# ((1,), (0,))  # or ((1,), (0,))
```

## What the gate checks

The gate checks whether the predicted axes match the reference solution, which is computed by analyzing the shapes of the input arrays and applying the broadcasting rules. The reference solution uses NumPy's `broadcast_to` function to determine the output shape of the broadcasted operation and then predicts the axes to sum based on the input shapes. The gate requires an exact match between the predicted axes and the reference solution.
