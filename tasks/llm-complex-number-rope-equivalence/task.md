## Context

Rotary Position Embedding (RoPE) injects relative positional information into token embeddings by rotating pairs of dimensions in the complex plane.  
For a pair $(x_{2i}, x_{2i+1})$ and an angle $\theta$, RoPE applies the rotation matrix  

$$
\begin{bmatrix}
\cos \theta & -\sin \theta\\[4pt]
\sin \theta & \phantom{-}\cos \theta
\end{bmatrix},
$$

which is equivalent to multiplying the complex number $z = x_{2i} + i\,x_{2i+1}$ by $\exp(i\theta)$.  
The angle depends on the token position and a frequency schedule:  

$$
\theta_{p,i} = \frac{p}{10000^{\,i/(d/2)}},
$$

where $p$ is the position index, $i$ indexes the dimension pair, and $d$ is the embedding dimensionality.

## Task

Implement the function `rope_complex(x, pos)` that applies RoPE to a batch of embeddings using complex multiplication.  

```python
def rope_complex(x: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """
    Apply Rotary Position Embedding (RoPE) via complex multiplication.

    Parameters
    ----------
    x : np.ndarray
        Input tensor of shape (batch, seq_len, dim).  `dim` must be even.
    pos : np.ndarray
        1‑D array of length `seq_len` containing the position indices for each token.

    Returns
    -------
    out : np.ndarray
        Tensor of the same shape as `x`, with RoPE applied to every pair of dimensions.
    """
```

The implementation must be fully vectorized, use only NumPy operations, and return a `float64` array. No Python loops are allowed.

## Example

```python
import numpy as np

# 2‑D embeddings: batch=1, seq_len=3, dim=4
x = np.array([[[0., 0., 0., 0.],
               [1., 0., 0., 0.],
               [0., 1., 0., 0.]]], dtype=np.float64)

pos = np.arange(3)  # positions 0, 1, 2

D = rope_complex(x, pos)
print(D)
```

Output (values rounded for readability):

```
[[[ 0.          0.          0.          0.        ]
  [ 0.99500417 -0.09983342  0.          0.        ]
  [ 0.         -1.          0.          0.        ]]]
```

## What the gate checks

The grader computes a reference RoPE using the standard rotation‑matrix formulation and compares it to your output with the metric `max_abs_err`.  
Your implementation must satisfy  

$$
\max_{i,j,k} |\, \text{candidate}_{ijk} - \text{reference}_{ijk}\,| \le 10^{-7}.
$$

If this bound is exceeded or an exception occurs during evaluation, the gate fails.
