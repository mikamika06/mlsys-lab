## Context

In language‑model inference, the raw logits $x_{i,t}$ for token $t$ in position $i$ are often adjusted by a *bias map* that adds a fixed value to selected tokens.  
If a bias map $\mathcal{B}$ contains an entry $(t,b_t)$, the corrected logit is

$$
y_{i,t} = x_{i,t} + b_t,
$$

otherwise $y_{i,t}=x_{i,t}$.  The bias values are independent of the position and apply to every row in the logits matrix.

## Task

Implement a function that applies such a bias map to a batch of logits:

```python
def apply_logit_bias_map(logits: np.ndarray, bias_map: dict[int, float]) -> np.ndarray:
    ...
```

`logits` is a 2‑D NumPy array of shape $(n,\;d)$ where $n$ is the number of positions and $d$ the vocabulary size.  
`bias_map` maps integer token indices to floating point bias values. The function must return a new array of the same shape with the biases added as described above.

## Example

```python
import numpy as np

logits = np.array([[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]])
bias_map = {0: -0.05, 2: 0.10}

biased = apply_logit_bias_map(logits, bias_map)
print(biased)
# [[ 0.05  0.20  0.40]
#  [ 0.35  0.50  0.70]]
```

## What the gate checks

The grader computes a reference implementation using NumPy and compares your output with it.  
It reports the maximum absolute difference

$$
\max_{i,t} |\, \hat y_{i,t} - y_{i,t}\,|
$$

between your result $\hat y$ and the reference $y$.  The solution must achieve a value **≤ $10^{-7}$**.  Any deviation larger than this threshold will cause the gate to fail.
