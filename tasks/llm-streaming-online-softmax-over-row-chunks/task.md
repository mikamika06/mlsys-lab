## Context

The softmax function converts a vector of logits $x \in \mathbb{R}^n$ into probabilities:

$$
\mathrm{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}} .
$$

A direct implementation can overflow when logits contain large values. The numerically stable form uses the log-sum-exp trick:

$$
\mathrm{softmax}(x_i) = \frac{e^{x_i-m}}{\sum_j e^{x_j-m}},
$$

where $m = \max_j x_j$.

For very long rows, storing or processing the entire row at once may be undesirable. The online softmax algorithm processes chunks while maintaining a running maximum $m$ and running exponential sum $l$. When a new chunk has maximum $m_c$, the state update is:

$$
m_{\mathrm{new}} = \max(m, m_c)
$$

and

$$
l_{\mathrm{new}} =
l e^{m-m_{\mathrm{new}}}
+
\sum_{x_i \in c} e^{x_i-m_{\mathrm{new}}}.
$$

After all chunks are processed, probabilities can be reconstructed as:

$$
p_i = \frac{e^{x_i-m}}{l}.
$$

## Task

Implement `stream_softmax_row_chunks(logits, chunk_size)`.

The function receives a 2-D NumPy array `logits` of shape $(r, n)$ and an integer `chunk_size`. It returns a NumPy array of the same shape containing the softmax value of every row.

The implementation must compute each row by processing columns in chunks. It should maintain the online softmax state instead of applying softmax independently to each chunk.

The output must be `float64`.

```python
def stream_softmax_row_chunks(
    logits: np.ndarray,
    chunk_size: int
) -> np.ndarray:
    ...
```

## Example

```python
import numpy as np

x = np.array([[1.0, 2.0, 3.0, 4.0]])
y = stream_softmax_row_chunks(x, 2)

# approximately:
# [[0.0320586, 0.0871443, 0.2368828, 0.6439143]]
```

## What the gate checks

The gate compares the returned probabilities against a NumPy reference implementation of full stable softmax. The maximum absolute difference

$$
\max_i |p_i - \hat{p}_i|
$$

must be less than $10^{-6}$.

The reference is computed from the logits during grading. Implementations that normalize each chunk separately will produce incorrect probabilities and fail.
