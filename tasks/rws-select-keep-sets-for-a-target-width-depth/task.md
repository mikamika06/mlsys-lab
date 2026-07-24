## Context

In many neural‑network compression pipelines a *width budget* and a *depth budget* are imposed simultaneously.  
Given two importance vectors  

$$w \in \mathbb{R}^{n_w},\qquad l \in \mathbb{R}^{n_l}$$  

representing the relative worth of each width candidate and each layer, we wish to keep exactly $d_{\text{target}}$ widths and $L_{\text{target}}$ layers.  
The natural greedy strategy is to select the indices with the largest importance values.

## Task

Implement `select_keep_sets`:

```python
def select_keep_sets(width_importance: np.ndarray,
                     layer_importance: np.ndarray,
                     d_target: int,
                     L_target: int) -> tuple[np.ndarray, np.ndarray]:
    ...
```

The function receives two 1‑D NumPy arrays of floats and two integers.  
It must return a pair `(keep_widths, keep_layers)` where each element is a 1‑D `np.ndarray` of dtype `int64`.  
Each array contains the indices (0‑based) of the selected widths or layers, sorted in descending order of importance.

## Example

```python
import numpy as np
w = np.array([0.2, 0.5, 0.1, 0.9])
l = np.array([0.3, 0.4, 0.8])
keep_w, keep_l = select_keep_sets(w, l, d_target=2, L_target=1)
print(keep_w)   # [3 1]
print(keep_l)   # [2]
```

## What the gate checks

The grader computes a reference solution by sorting each importance vector in descending order and taking the first `d_target` or `L_target` indices.  
Your output must match this reference exactly; otherwise the `exact_match` metric will be 0.0.
