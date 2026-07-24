## Context

The softmax transformation converts logits $x \in \mathbb{R}^d$ into a probability distribution:

$$
\mathrm{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}} .
$$

A direct implementation can overflow when logits contain large positive values. The stable form subtracts the maximum logit:

$$
\mathrm{softmax}(x_i) =
\frac{e^{x_i-m}}{\sum_j e^{x_j-m}},
\qquad
m = \max_j x_j .
$$

The subtraction keeps the exponentials in a representable range while preserving the mathematical result.

This task also models deterministic memory behaviour. A kernel emits a byte-address access trace. The grader sends that trace through a fixed cache simulator. The simulator output is the oracle for cache behaviour; no wall-clock timing or machine-specific measurements are used.

## Task

Implement `stable_softmax_kernel(logits)`:

```python
def stable_softmax_kernel(logits: np.ndarray) -> tuple[np.ndarray, list[int]]:
    ...
```

The input is a 2-D NumPy array of shape $(n, d)$ containing float64 logits. Return:

1. A float64 array of shape $(n, d)$ containing the row-wise stable softmax probabilities.
2. A list of integer byte addresses describing the memory access trace.

The trace must follow the deterministic row-major traversal used by the reference. It records contiguous float64 element addresses for reading the input and writing the output.

## Example

```python
import numpy as np

x = np.array([[1000.0, 1001.0, 1002.0],
              [1.0, 2.0, 3.0]])

y, trace = stable_softmax_kernel(x)

# y is approximately:
# [[0.09003057, 0.24472847, 0.66524096],
#  [0.09003057, 0.24472847, 0.66524096]]
```

## What the gate checks

The grader computes the reference probabilities using float64 max-subtraction softmax and measures mean KL divergence. The submitted output must satisfy

$$
\mathrm{mean\_kl} \le 10^{-12}.
$$

The grader also computes deterministic cache results for the submitted trace and the reference trace. The cache behaviour score is $1$ only when the simulator reports identical miss behaviour for both traces.
