## Context

Softmax converts logits into a probability distribution. For a vector
$x \in \mathbb{R}^n$, it is defined as

$$
\operatorname{softmax}(x_i) =
\frac{e^{x_i}}{\sum_j e^{x_j}} .
$$

Stable implementations subtract the maximum value before exponentiation:

$$
\operatorname{softmax}(x_i) =
\frac{e^{x_i-m}}{\sum_j e^{x_j-m}},
\qquad m = \max_j x_j .
$$

In attention systems, logits can have large magnitudes. Computing the
exponentials and reduction directly in float16 can lose precision because the
format has limited range and mantissa bits. Production kernels often upcast the
values to float32 for the reduction and exponentiation, then return the result
in float16 or float32 as needed.

The reference computation uses float64 arithmetic:

$$
p_i =
\frac{\exp(x_i-\max(x))}
{\sum_j \exp(x_j-\max(x))}.
$$

The implementation in this task should follow the common mixed-precision
strategy: accept float16 inputs, perform the softmax calculation using float32
intermediate values, and return float32 probabilities.

## Task

Implement `softmax_fp32(x)`:

```python
def softmax_fp32(x: np.ndarray) -> np.ndarray:
    ...
```

The input is a two-dimensional NumPy array of float16 logits. Apply softmax
independently to each row. Convert to float32 before the maximum, subtraction,
exponential, and summation operations. Return a float32 array with the same
shape.

Do not use Python loops over rows. Use NumPy operations that broadcast across
the batch dimension.

## Example

```python
import numpy as np

x = np.array([[10, 11, 12]], dtype=np.float16)
p = softmax_fp32(x)

# p is approximately:
# [[0.0900, 0.2447, 0.6652]]
# dtype is float32
```

## What the gate checks

The gate creates an adversarial float16 input with large-magnitude logits. It
computes a float64 NumPy oracle and compares the submitted result against it
using maximum absolute error:

$$
\max_i |p_i^{candidate} - p_i^{oracle}|.
$$

The submitted implementation must have error at most $10^{-3}$. The grader also
computes a pure-float16 softmax baseline and verifies that its error is strictly
larger than the required fp32-accumulated path, demonstrating the precision
difference being tested.
