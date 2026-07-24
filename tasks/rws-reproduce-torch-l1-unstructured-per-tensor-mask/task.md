## Context

In neural network pruning, *unstructured* L1 pruning removes a fraction of the smallest‑magnitude weights. For a weight tensor $W \in \mathbb{R}^{n_1\times\dots\times n_k}$ we flatten it to a vector $\mathbf{w} = \operatorname{vec}(W)$ and choose a threshold $\tau$ such that exactly an amount $s\in[0,1]$ of the elements satisfy $|w_i| < \tau$. The binary mask $M$ is then

$$
M_{i}= \begin{cases}
0 & |w_i| < \tau\\
1 & \text{otherwise.}
\end{cases}
$$

PyTorch’s `torch.nn.utils.prune.l1_unstructured` implements this by sorting the absolute values and zeroing the lowest $s$ fraction.

## Task

Implement `l1_unstructured_mask(weight, amount)`:

```python
def l1_unstructured_mask(weight: np.ndarray, amount: float) -> np.ndarray:
    ...
```

The function receives a NumPy array `weight` of arbitrary shape and a float `amount` in $[0,1]$. It must return a boolean mask of the same shape where exactly $\lfloor \text{amount}\times\text{numel}(\text{weight})\rfloor$ entries are zeroed. The algorithm should be fully vectorised; no Python loops.

## Example

```python
import numpy as np
W = np.array([[0.1, -0.5], [2.0, 0.3]])
mask = l1_unstructured_mask(W, amount=0.25)
# mask is
# array([[False, True],
#        [True , True]], dtype=bool)
```

Here the flattened weights are `[0.1, -0.5, 2.0, 0.3]`. The smallest magnitude is `0.1`; with amount = 0.25 we zero exactly one element.

## What the gate checks

The grader computes a reference mask using NumPy’s sorting functions and compares it to the candidate mask element‑wise. The metric `exact_match` must equal 1.0 for the submission to pass.
