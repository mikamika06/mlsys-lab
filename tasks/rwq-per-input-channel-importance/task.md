## Context

In many quantisation pipelines the importance of each input channel is estimated by the average magnitude of its activations across tokens.  
For a tensor $X \in \mathbb{R}^{B\times T\times C}$, where $B$ is batch size, $T$ token length and $C$ the number of channels, the per‑channel importance vector is

$$
s_c = \frac{1}{BT}\sum_{b=1}^B\sum_{t=1}^T |X_{b,t,c}|\,,
$$

which can be computed efficiently with Python by averaging over the first two axes.

## Task

Implement `per_input_channel_importance`:

```python
def per_input_channel_importance(X: list[list[list[float]]]) -> list[float]:
    ...
```

The function receives a 3‑D list of shape `(B, T, C)` and must return a 1‑D array of length `C`.  
All computations should use vectorised Python only; no explicit Python loops. The result must be of type `float64`.

## Example

```python
X = [
    [[1, -2], [3, 4]],
    [[-5, 6], [7, -8]]
]   # shape (2, 2, 2)

s = per_input_channel_importance(X)
print(s)  # [4.0, 5.0]
```

The first channel has mean absolute value $(|1|+|3|+|-5|+|7|)/(2\cdot2)=3$, the second channel $ (| -2 | + |4| + |6| + |-8|)/(2\cdot2)=4$.

## What the gate checks

The grader generates a deterministic tensor and computes the exact Python reference:

$$
s_{\text{ref}} = \operatorname{mean}\bigl(|X|\bigr)\quad\text{over axes }(0,1).
$$

Your implementation is compared to this reference using the global relative error metric from `arena.scorers.rel_err`.  
The gate requires $\mathrm{rel\_err} \le 10^{-8}$.
