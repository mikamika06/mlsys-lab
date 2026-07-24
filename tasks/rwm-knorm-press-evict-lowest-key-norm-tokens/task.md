## Context

In many transformer‑style models a key–value cache stores token embeddings as pairs $(k_i, v_i)$.  
When the cache reaches its capacity $B$, a policy decides which tokens to evict.  
A simple yet effective strategy is to keep only the $B$ keys with largest Euclidean norm
$$\lVert k_i \rVert_2 = \sqrt{\sum_{j=1}^{d} k_{ij}^2}\,,$$
and discard the rest.  This policy is called **knorm press**.

## Task

Implement `knorm_press(keys, values, capacity)`:

```python
def knorm_press(keys: np.ndarray,
                values: np.ndarray,
                capacity: int) -> tuple[np.ndarray, np.ndarray]:
    ...
```

* `keys` – a 2‑D NumPy array of shape $(n,d)$ containing $n$ key vectors.  
* `values` – a 2‑D NumPy array of shape $(n,v)$ containing the corresponding values.  
* `capacity` – an integer $B \ge 0$.  

The function must return two arrays `(kept_keys, kept_values)` that contain exactly the
top–$B$ tokens by key norm, preserving their original relative order.
If $n \le B$, all tokens are returned; if $B = 0$, empty arrays of shape $(0,d)$ and $(0,v)$ should be returned.

The output arrays must have dtype `float64`.

## Example

```python
import numpy as np
keys   = np.array([[1, 0], [0, 2], [3, 4]], dtype=np.float64)
values = np.arange(9).reshape(3, 3)          # arbitrary values
kept_k, kept_v = knorm_press(keys, values, capacity=2)

# keys sorted by norm: (3,4), (0,2), (1,0)
# top-2 are indices 2 and 1 in original order -> [2,1]
print(kept_k)   # [[3. 4.]
                #  [0. 2.]]
print(kept_v)   # [[6 7 8]
                #  [3 4 5]]
```

## What the gate checks

The grader computes a NumPy oracle that selects the top‑$B$ keys by Euclidean norm
and verifies that the candidate’s output matches this reference exactly.
No other metrics are required. The gate passes only if the returned arrays are
identical to the oracle’s arrays in shape, dtype and values.
