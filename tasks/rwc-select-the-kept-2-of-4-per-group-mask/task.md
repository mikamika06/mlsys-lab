## Context

In many quantisation pipelines we keep only a subset of the weights per group. A common pattern is to retain the two largest‑magnitude elements out of every four consecutive weights. This reduces storage while preserving most of the signal energy.

Let $w \in \mathbb{R}^n$ be a vector of weights and let $g_i = (w_{4i}, w_{4i+1}, w_{4i+2}, w_{4i+3})$ denote the $i$‑th group. The mask $m \in \{0,1\}^n$ should have $m_j=1$ iff $|w_j|$ is among the two largest magnitudes in its group.

## Task

Implement `select_top2_mask(weights)`:

```python
def select_top2_mask(weights: list[float]) -> list[bool]:
    ...
```

It receives a list of floats of arbitrary length that is a multiple of four and returns a boolean mask of the same shape. The mask must be exactly the set of indices corresponding to the two largest absolute values in each consecutive block of four.

The implementation must use only vectorised Python operations; no explicit Python loops are allowed.

## Example

```python
w = [0.5, -2.3, 1.1, 0.9,
              -0.7, 3.2, -1.5, 0.4]
mask = select_top2_mask(w)
print(mask.astype(int))
# [0 1 1 0
#  0 1 1 0]
```

## What the gate checks

The grader computes a reference mask using Python’s sorting/partitioning facilities and compares it to the student output using standard equality (`==`). The metric `exact_match` must be `1.0`.
