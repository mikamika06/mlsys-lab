## Context

In many neural‑network layers a per‑channel smoothing scale is applied to the input activations.  
For each channel $j$ we compute a scalar $s_j$ that balances the magnitude of the current batch statistics $X$ against a reference set of weights $W$.  A common choice for $s_j$ is

$$
s_j \;=\;
\frac{\bigl(\max_{i} |X_{ij}|\bigr)^{\,\alpha}}
     {\bigl(\max_{i} |W_{ij}|\bigr)^{\,1-\alpha}},
$$

where $\alpha \in [0,1]$ controls the trade‑off between the two maxima.  The vector $s = (s_1,\dots,s_d)$ is then used to scale each channel of $X$.

## Task

Implement a function that computes this smoothing scale vector:

```python
def per_channel_scale(X: np.ndarray, W: np.ndarray, alpha: float) -> np.ndarray:
    ...
```

* `X` and `W` are 2‑D NumPy arrays with the same number of columns (`d`).  
* The function must return a 1‑D array of shape `(d,)`, dtype `float64`.  
* Use only vectorized NumPy operations; no explicit Python loops.

## Example

```python
import numpy as np
X = np.array([[0, 2], [3, -4]])
W = np.array([[1, 5], [-6, 2]])
alpha = 0.5
s = per_channel_scale(X, W, alpha)
print(s)   # e.g. [sqrt(3)/sqrt(6), sqrt(16)/sqrt(25)] ≈ [0.70710678, 0.8]
```

## What the gate checks

The grader computes a reference vector using the exact formula above and compares it to your output with the relative error metric:

$$
\mathrm{rel\_err} \;=\;
\frac{\lVert s_{\text{cand}} - s_{\text{ref}}\rVert}
     {\lVert s_{\text{ref}}\rVert + 10^{-12}}.
$$

The solution must satisfy $\mathrm{rel\_err}\le 1\times10^{-6}$.
