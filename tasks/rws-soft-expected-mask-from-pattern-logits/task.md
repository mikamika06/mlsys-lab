## Context

In many machine‑learning pipelines a set of *patterns* is weighted by the probability distribution produced from a set of logits.  
Let $L \in \mathbb{R}^{B\times P}$ be a batch of logits, where $B$ is the number of examples and $P$ the number of patterns.  The softmax over each row gives a probability vector

$$
s_{b,p} = \frac{\exp(L_{b,p})}{\sum_{q=1}^{P}\exp(L_{b,q})},
$$

which we denote by $S$.  
Let $M \in \mathbb{R}^{P\times D}$ be the pattern matrix, each row being a *mask* of dimension $D$.  The **soft expected mask** is then defined as

$$
E = S\, M,
$$

i.e. for every example we take the weighted sum of all patterns according to their softmax probabilities.

This operation appears in attention‑like modules and in probabilistic mixture models where a continuous output is obtained by averaging discrete prototypes.

## Task

Implement the function `soft_expected_mask`:

```python
def soft_expected_mask(logits: np.ndarray, patterns: np.ndarray) -> np.ndarray:
    ...
```

* `logits` – a 2‑D NumPy array of shape `(B, P)` containing arbitrary real numbers.
* `patterns` – a 2‑D NumPy array of shape `(P, D)` containing the pattern masks.

The function must return a NumPy array of shape `(B, D)` with dtype `float64`.  
Use only vectorised NumPy operations; no explicit Python loops are allowed.

## Example

```python
import numpy as np

logits   = np.array([[0.0, 1.0], [2.0, -1.0]])
patterns = np.array([[1.0, 0.0],
                     [0.0, 1.0]])

mask = soft_expected_mask(logits, patterns)
print(mask)
# [[0.73105858 0.26894142]
#  [0.88079708 0.11920292]]
```

The first row is the softmax of `[0,1]` times the identity pattern matrix; the second row uses the softmax of `[2,-1]`.

## What the gate checks

The grader computes a NumPy reference `E_ref = softmax(logits) @ patterns`.  
Your implementation must produce an array whose global relative L2 error satisfies

$$
\mathrm{rel\_err} = \frac{\lVert E_{\text{cand}} - E_{\text{ref}}\rVert}{\lVert E_{\text{ref}}\rVert}
\le 10^{-6}.
$$

The gate will fail if the relative error exceeds this threshold.
