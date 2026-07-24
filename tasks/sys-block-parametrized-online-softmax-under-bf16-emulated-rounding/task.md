## Context

Softmax converts a vector of logits $x \in \mathbb{R}^n$ into a probability vector:

$$
\operatorname{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}.
$$

A numerically stable implementation tracks a running maximum. For a stream of blocks,
the running state after processing values is a pair $(m, l)$ where $m$ is the current
maximum and $l$ is the scaled exponential sum. When a new block maximum $m_b$ is
seen, the state is updated with

$$
m' = \max(m, m_b),
$$

$$
l' = l e^{m-m'} + \sum_{x_i \in \text{block}} e^{x_i-m'}.
$$

The final probabilities are obtained by applying the accumulated state to every
element:

$$
p_i = \frac{e^{x_i-m}}{l}.
$$

In this task, arithmetic is performed with a simulated bfloat16 format. The helper
operation rounds every intermediate value to bf16 precision using round-to-nearest-even
behavior. This emulates reduced precision while keeping the implementation in NumPy.

## Task

Implement `tiled_online_softmax(x, B)`:

```python
def tiled_online_softmax(x: np.ndarray, B: int) -> np.ndarray:
    ...
```

The input `x` is a one-dimensional NumPy array of logits. `B` is the block size and
will be one of $16$, $64$, or $256$. Process the input in consecutive blocks of size
$B$ using the online softmax recurrence.

All intermediate floating point values used by the algorithm must be rounded through
the bf16 emulation rule. Return a NumPy array of dtype `float64` containing the final
softmax values.

Do not use Python libraries that provide a complete softmax implementation.

## Example

```python
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0])
y = tiled_online_softmax(x, 2)

# y is close to:
# [0.0321, 0.0871, 0.2369, 0.6439]
```

## What the gate checks

The gate computes an independent NumPy reference implementation of the same online
recurrence with explicit bf16-emulated rounding. It tests several generated logit
vectors and block sizes $B \in \{16,64,256\}$.

The reported metric is

$$
\operatorname{max\_abs\_err} =
\max_i |y_i-\hat{y}_i|.
$$

The implementation passes when this error is below $10^{-3}$.
