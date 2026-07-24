## Context

In language modeling, a model outputs a vector of logits $\mathbf{z} \in \mathbb{R}^{V}$ for each position in the sequence, where $V$ is the vocabulary size.  
The greedy decoding strategy selects the token with the largest logit value:

$$\hat{y} = \arg\max_{i=1,\dots,V} z_i.$$

This choice corresponds to a deterministic next‑token prediction that maximises the model’s confidence.

## Task

Implement `greedy_argmax_next_token` that takes a 2‑D NumPy array of logits with shape $(B, V)$ and returns a 1‑D array of length $B$ containing the index of the maximum logit for each batch element.

```python
def greedy_argmax_next_token(logits: np.ndarray) -> np.ndarray:
    ...
```

The function must use NumPy only; no Python loops are allowed. The output should be a NumPy array of dtype `int64`.

## Example

```python
import numpy as np
logits = np.array([[0.1, 2.3, -1.5],
                   [4.2, 0.0, 1.1]])
indices = greedy_argmax_next_token(logits)
print(indices)   # [1 0]
```

## What the gate checks

The grader generates random batches of logits and compares your output to NumPy’s `argmax`.  
A single mismatch or a shape/typing error causes the gate to fail. The metric used is `exact_match`, which must equal `1.0` for success.
