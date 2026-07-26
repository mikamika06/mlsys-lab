## Context

In many quantisation schemes, such as AWQ, a per‑channel scale vector is derived from the activations of a neural network layer.  
For an activation tensor $X \in \mathbb{R}^{B\times T\times C}$ where $B$ is batch size, $T$ is sequence length and $C$ is the number of channels, the *activation‑scale statistic* for channel $j$ is defined as

$$
s_j = \frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T} |X_{b,t,j}| .
$$

The vector $\mathbf{s} = (s_0,\dots,s_{C-1})^\top$ captures the average magnitude of each channel and is used to normalise weights before quantisation.

## Task

Implement `compute_activation_scale`:

```python
def compute_activation_scale(X: np.ndarray) -> np.ndarray:
    ...
```

The function receives a 3‑D NumPy array of shape `(batch, seq_len, channels)` and must return a 1‑D float64 array of length `channels`.  
Use only vectorised NumPy operations; no explicit Python loops.

## Example

```python
import numpy as np
X = np.array([
    [[-1.0, 2.0], [3.0, -4.0]],
    [[5.0, -6.0], [-7.0, 8.0]]
])  # shape (2, 2, 2)
s = compute_activation_scale(X)
print(s)   # array([4., 5.])
```

Explanation:  
For channel 0 the values are -1, 3, 5, -7: `(1+3+5+7)/(2*2) = 16/4 = 4.0`.  
For channel 1: `(|2|+|-4|+|-6|+|8|)/(2*2)=20/4=5.0`.  
Thus the output should be `[4.0, 5.0]`.

## What the gate checks

The grader computes a reference statistic with NumPy and compares it to your result using the metric `max_abs_err`, which is the maximum absolute difference between corresponding elements. The solution must satisfy

$$
\max_j |\, \hat{s}_j - s_j \,| \le 10^{-6}.
$$

The gate also verifies that the output shape matches `(channels,)`.
