## Context

In quantized neural networks, it is common to apply a separate scaling factor per output channel of the weight matrix and a separate scale per token (row) of the activation tensor. The weight‑channel scales are typically chosen as the Euclidean norm of each row of the weight matrix $W \in \mathbb{R}^{C_{\text{out}}\times C_{\text{in}}}$, while the activation scales are the norms of each input vector $x_i \in \mathbb{R}^{C_{\text{in}}}$. These scalars are later used to rescale quantized tensors so that the dynamic range is preserved.

## Task

Implement `torchao_scale_granularities(W, X)`:

```python
def torchao_scale_granularities(W: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ...
```

It receives:
- `W`: a 2‑D NumPy array of shape `(out_channels, in_features)`
- `X`: a 2‑D NumPy array of shape `(batch_size, in_features)`

and must return a tuple
`(weight_scales, activation_scales)` where

* `weight_scales[i] = \lVert W_i\rVert_2`, the Euclidean norm of row $i$ of $W$.
* `activation_scales[j] = \lVert X_j\rVert_2`, the Euclidean norm of row $j$ of $X`.

The implementation must use only vectorised NumPy operations; no explicit Python loops are allowed.

## Example

```python
import numpy as np
W = np.array([[1, 0], [0, 3]])
X = np.array([[4, 5], [6, 7]])
weight_scales, activation_scales = torchao_scale_granularities(W, X)
print(weight_scales)      # [1. 3.]
print(activation_scales)  # [6.40312424 9.16515139]
```

## What the gate checks

The grader computes a reference implementation using NumPy and compares your output against it with the global relative L2 error

$$
\mathrm{rel\_err} = \frac{\lVert y_{\text{sol}} - y_{\text{ref}}\rVert}{\lVert y_{\text{ref}}\rVert},
$$

where $y$ is the concatenation of the two scale vectors. The solution must satisfy $\mathrm{rel\_err}\le 10^{-8}$.
