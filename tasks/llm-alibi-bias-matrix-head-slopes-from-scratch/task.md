## Context

The Attention Linear Bias (ALiBi) scheme augments the attention logits with a linear bias that depends only on the relative distance between query and key positions.  
For a sequence of length $L$ and a head with slope $s$, the bias for a pair of positions $(i,j)$ is

$$b_{ij} = -\, s \, (i-j).$$

The negative sign ensures that tokens farther away receive a larger penalty.  When multiple heads are present, each head has its own slope $s_h$ and therefore its own bias matrix.

## Task

Implement `alibi_bias_matrix`:

```python
def alibi_bias_matrix(head_slopes: np.ndarray, seq_len: int) -> np.ndarray:
    ...
```

The function receives a 1‑D NumPy array of shape $(H)$ containing the slopes for $H$ heads and an integer `seq_len`.  
It must return a 3‑D float64 array of shape $(H,\;L,\;L)$ where

$$\text{bias}[h,i,j] = -\, \texttt{head\_slopes}[h]\,(i-j).$$

The implementation should be fully vectorised with NumPy only – no Python loops.

## Example

```python
import numpy as np
slopes = np.array([0.01, 0.02])
biases = alibi_bias_matrix(slopes, 4)
print(biases.shape)          # (2, 4, 4)
print(np.round(biases[0], 3))
# [[ 0.   -0.01 -0.02 -0.03]
#  [ 0.01  0.   -0.01 -0.02]
#  [ 0.02  0.01  0.   -0.01]
#  [ 0.03  0.02  0.01  0. ]]
```

## What the gate checks

The grader recomputes the reference bias matrix using the same formula and compares it to your output with the metric `max_abs_err`.  
Your solution must satisfy

$$\max_{h,i,j}\,|\,\text{bias}_{hij}^{\text{your}} - \text{bias}_{hij}^{\text{ref}}\;|\;\le 10^{-6}.$$

A correct vectorised implementation will pass this gate. A wrong sign or use of absolute distance will fail.
