## Context

In reinforcement learning a *student* policy is updated using two kinds of data:
on‑policy samples that come from the current policy roll‑out, and off‑policy samples
that are drawn from an external dataset.  
When computing a loss we must route each token to the correct target set,
otherwise the gradients will be corrupted.

Let $t_i$ denote the $i$‑th token in a batch of size $m$.  
A boolean array $\texttt{tags}$ of length $m$ indicates whether $t_i$
belongs to the on‑policy group ($\texttt{True}$) or the off‑policy group
($\texttt{False}$).  
Two target tensors are supplied:

* $\mathbf{O}\in \mathbb{R}^{n_{\text{on}}\times d}$ – targets for on‑policy tokens,
* $\mathbf{F}\in \mathbb{R}^{n_{\text{off}}\times d}$ – targets for off‑policy tokens.

The number of tokens satisfies  
$$m = n_{\text{on}} + n_{\text{off}}.$$

Your task is to produce, for each token $t_i$, the target vector that
corresponds to its group.  The output must preserve the original order of
tokens and be a NumPy array of dtype `float64`.

## Task

Implement `route_and_apply`:

```python
def route_and_apply(tags: np.ndarray,
                    on_policy_targets: np.ndarray,
                    off_policy_targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ...
```

* `tags`: 1‑D boolean array of length $m$.
* `on_policy_targets`: 2‑D float64 array of shape $(n_{\text{on}}, d)$.
* `off_policy_targets`: 2‑D float64 array of shape $(n_{\text{off}}, d)$.

Return a tuple `(routing_mask, routed_targets)` where

* `routing_mask` is a copy of `tags`.
* `routed_targets` is an array of shape $(m,d)$ such that
  * for every index $i$ with `tags[i]==True`,  
    `routed_targets[i] == on_policy_targets[k]` where $k$
    counts the number of preceding `True` tags.
  * for every index $i$ with `tags[i]==False`,  
    `routed_targets[i] == off_policy_targets[l]` where $l$
    counts the number of preceding `False` tags.

The implementation must use only NumPy operations and no Python loops.

## Example

```python
import numpy as np

tags = np.array([True, False, True])
on_policy_targets = np.array([[1., 2.],
                              [3., 4.]])          # n_on = 2
off_policy_targets = np.array([[5., 6.]])        # n_off = 1

routing_mask, routed_targets = route_and_apply(tags,
                                               on_policy_targets,
                                               off_policy_targets)

print(routing_mask)
# [ True False  True]

print(routed_targets)
# [[1. 2.]
#  [5. 6.]
#  [3. 4.]]
```

## What the gate checks

The grader recomputes the routing using a NumPy reference implementation
and compares your output to it with `np.array_equal`.  
Both the mask and the routed targets must match exactly; otherwise the
gate fails.
