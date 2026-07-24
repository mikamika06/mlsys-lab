## Context

Rotary Position Embedding (RoPE) encodes the position of tokens in a sequence by applying sinusoidal functions to each dimension. For a position $p$ and embedding dimension $d$, RoPE defines a set of frequencies  

$$f_i = \frac{1}{10000^{\,i/d}},\qquad i=0,\dots,d-1,$$

and computes the angle for each frequency  

$$\theta_{p,i} = p \cdot f_i.$$

The embedding is then given by $\sin(\theta_{p,i})$ (often paired with a cosine term).  
When extending a model trained on sequences of length $L_{\text{train}}$ to longer target lengths $L_{\text{target}}$, it is common to linearly interpolate the positions:

$$\tilde p = p \cdot \frac{L_{\text{target}}}{L_{\text{train}}}.$$

The goal of this task is to implement a vectorised routine that applies this linear interpolation and then computes the RoPE sine embeddings.

## Task

Implement `linear_rope`:

```python
def linear_rope(positions: np.ndarray, dim: int,
                L_train: int, L_target: int) -> np.ndarray:
    ...
```

* `positions`: 1‑D array of integer or float positions (length $n$).  
* `dim`: embedding dimension ($d$).  
* `L_train`, `L_target`: training and target sequence lengths.  

The function should return a NumPy array of shape `(n, dim)` containing the sine embeddings for each interpolated position. The implementation must be fully vectorised (no Python loops) and use only NumPy operations.

## Example

```python
import numpy as np
positions = np.array([0, 1, 2])
dim = 4
L_train = 10
L_target = 20
emb = linear_rope(positions, dim, L_train, L_target)
print(emb.shape)   # (3, 4)
```

## What the gate checks

The grader computes a reference array using NumPy and compares it to your output with the metric  

$$\max_{i,j} |\, \text{your}[i,j] - \text{reference}[i,j] \,|.$$

Your solution must achieve `max_abs_err <= 1e-6`. Any deviation larger than this threshold will fail the gate.
