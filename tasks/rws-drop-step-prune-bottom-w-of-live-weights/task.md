## Context

In many neural‑network training pipelines a *drop step* is used to prune the smallest‑magnitude weights that are still considered *live*.  
Let $w \in \mathbb{R}^n$ be the weight vector and let $\mathbf m \in \{0,1\}^n$ be a binary mask where $m_i=1$ indicates that $w_i$ is currently live.  
Given a fraction $f\in[0,1]$, the drop step removes the bottom $f$‑fraction of the live weights by magnitude.

Formally, let
$$L = \{\, i \mid m_i = 1 \,\}$$
be the set of live indices and let $k = \lfloor |L| f \rfloor$.  
The algorithm selects the $k$ indices in $L$ with smallest $|w_i|$ and sets their mask entries to $0$.

## Task

Implement `drop_step_prune`:

```python
def drop_step_prune(weights: np.ndarray, mask: np.ndarray, drop_frac: float) -> np.ndarray:
    ...
```

The function receives a 1‑D NumPy array of weights, a boolean or integer mask of the same length, and a fraction `drop_frac`.  
It must return a new mask (NumPy array of dtype bool) where exactly the bottom `drop_frac` fraction of live weights have been pruned.  
If `drop_frac` is $0$ no weight should be removed; if it is $1$ all live weights are dropped.

The implementation must use only NumPy operations and run in $O(n \log n)$ time or better.

## Example

```python
import numpy as np
w = np.array([ 0.5, -1.2, 3.4, -0.7])
m = np.array([True, True, False, True])
# drop the bottom 50 % of live weights (two live indices)
new_m = drop_step_prune(w, m, 0.5)
print(new_m)   # [False  True False  True]
```

Here the live weights are $[0.5,\,-1.2,\,-0.7]$; their magnitudes are $[0.5,\,1.2,\,0.7]$.  
The smallest magnitude is $0.5$, so that weight is removed.

## What the gate checks

Two tests are run:

* **Exact mask match** – The returned mask must be identical to a NumPy oracle that implements the same algorithm.
* **Determinism** – The function should not modify its inputs; it must return a new array.

The grader uses `np.array_equal` on the produced mask and the oracle’s mask.  
Any deviation, including changing the input mask or dropping an incorrect number of weights, causes the gate to fail.
