## Context

In a transformer layer the input token representation $x_{\text{in}} \in \mathbb{R}^d$ is transformed into an output representation $x_{\text{out}}\in\mathbb{R}^d$.  
The cosine similarity between two vectors is defined as

$$
\cos(x_{\text{in}}, x_{\text{out}}) = 
\frac{x_{\text{in}}\cdot x_{\text{out}}}
{\lVert x_{\text{in}}\rVert\,\lVert x_{\text{out}}\rVert},
$$

where $x_{\text{in}}\cdot x_{\text{out}}$ is the dot product and $\lVert \cdot\rVert$ denotes the Euclidean norm.  
For a batch of tokens we are interested in the mean cosine similarity over all token positions.

## Task

Implement `mean_cosine_similarity(in_states, out_states)`:

```python
def mean_cosine_similarity(in_states: np.ndarray,
                           out_states: np.ndarray) -> float:
    ...
```

`in_states` and `out_states` are 2‑D NumPy arrays of shape $(n_{\text{tokens}}, d)$ containing the input and output token representations from a single transformer block.  
Return the mean cosine similarity over all tokens as a Python `float`.  
The implementation must use only vectorised NumPy operations; no explicit Python loops.

## Example

```python
import numpy as np
in_states = np.array([[1, 0], [0, 1]])
out_states = np.array([[0.5, 0.5], [-0.5, 0.5]])
mean_cosine_similarity(in_states, out_states)
# ≈ 0.70710678
```

## What the gate checks

The grader generates several random test cases and compares your result to a NumPy reference implementation.  
It reports the maximum relative error

$$\mathrm{rel\_err} = \max_{\text{cases}}
\frac{|\,\hat y - y\,|}{|y| + 10^{-12}}.$$

The solution must satisfy $\mathrm{rel\_err}\le 1\times10^{-6}$.
