## Context

The angular distance between two non‑zero vectors $a,b\in \mathbb{R}^d$ is the angle $\theta$ between them, normalised to the interval $[0,1]$.  
It is defined as  

$$
\operatorname{angdist}(a,b)=\frac{\arccos(\cos(a,b))}{\pi},
$$

where the cosine similarity is  

$$
\cos(a,b)=\frac{a^\top b}{\lVert a\rVert\,\lVert b\rVert}.
$$

In deep learning libraries, one often needs to compute this distance for every pair of corresponding rows in two state tensors (e.g. activations or weight matrices) across several layers.  The result is typically a dictionary mapping layer names to arrays of angular distances.

## Task

Implement the function `angular_distance_per_layer`:

```python
def angular_distance_per_layer(states_a: dict[str, np.ndarray],
                               states_b: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    ...
```

* `states_a` and `states_b` are dictionaries that share the same keys.  
  Each value is a 2‑D NumPy array of shape `(n, d)` containing `n` vectors in `d` dimensions.
* For each key, compute the angular distance for every pair of corresponding rows
  (row *i* from `states_a[key]` with row *i* from `states_b[key]`) using the formula above.
* Return a dictionary mapping each key to a 1‑D NumPy array of shape `(n,)`.  
  All arrays must be of dtype `float64`.

The implementation must use only NumPy operations; no explicit Python loops over rows.

## Example

```python
import numpy as np

states_a = {
    "layer1": np.array([[1, 0], [0, 1]]),
    "layer2": np.array([[1, 1]])
}
states_b = {
    "layer1": np.array([[0, 1], [1, 0]]),
    "layer2": np.array([[-1, -1]])
}

distances = angular_distance_per_layer(states_a, states_b)
# distances["layer1"] ≈ array([0.5, 0.5])   # 90° / π
# distances["layer2"] ≈ array([1.0])       # 180° / π
```

## What the gate checks

The grader computes a reference implementation using NumPy and compares your output with it.
It reports the maximum absolute error over all returned values:

$$
\max_{k,i} \bigl|\,\text{your}(k,i)-\text{reference}(k,i)\,\bigr|.
$$

Your solution must achieve an error $\le 10^{-6}$.
