## Context

When quantizing neural‑network weights we often scale them by a factor $s>0$ and then round to the nearest integer:
$$\hat w = \operatorname{round}\!\left(\frac{w}{s}\right).$$
If during training a small correction vector $v$ is added to each weight, the rounded value may change.  
For every element $w_{ij}$ we compare

* the original rounding $\hat w_{ij}=\operatorname{round}(w_{ij}/s)$,
* the corrected rounding $\tilde w_{ij}=\operatorname{round}\!\bigl((w_{ij}+v_{ij})/s\bigr).$

We classify each element into one of three categories:

| label | meaning |
|-------|---------|
| $-1$  | $\tilde w < \hat w$ (rounded **down**) |
| $0$   | $\tilde w = \hat w$ (no change, **nearest**) |
| $+1$  | $\tilde w > \hat w$ (rounded **up**) |

The rounding rule used is NumPy’s `np.round`, which implements “round half to even”.

## Task

Implement the function

```python
def classify_rounding(W: np.ndarray, V: np.ndarray, s: float) -> np.ndarray:
    ...
```

* `W` – a 2‑D array of real weights (`float64`).
* `V` – a correction array of the same shape as `W`.
* `s` – positive scaling factor.

The function must return an integer array of the same shape as `W`, with values in $\{-1,0,+1\}$ according to the rule above.  
Use only NumPy operations; no explicit Python loops are required but allowed if you wish.

## Example

```python
import numpy as np

W = np.array([[ 0.2, -0.3],
              [ 1.5,  2.7]])
V = np.array([[ 0.05,  0.02],
              [-0.01,  0.03]])
s = 0.1

labels = classify_rounding(W, V, s)
print(labels)
# [[-1  0]
#  [ 1  0]]
```

Explanation:  
* For $w_{00}=0.2$, $\hat w=2$ and $\tilde w=\operatorname{round}(2.25)=2$ → equal (0).  
* For $w_{01}=-0.3$, $\hat w=-3$ and $\tilde w=\operatorname{round}(-2.98)=-3$ → equal (0).  
* For $w_{10}=1.5$, $\hat w=15$ and $\tilde w=\operatorname{round}(14.9)=15$ → equal (0).  
* For $w_{11}=2.7$, $\hat w=27$ and $\tilde w=\operatorname{round}(28.3)=28$ → up (+1).

## What the gate checks

The grader computes a NumPy reference implementation of the classification and compares it to your output using an exact match metric.  Your solution must produce exactly the same integer array for all test cases; otherwise the gate fails.
