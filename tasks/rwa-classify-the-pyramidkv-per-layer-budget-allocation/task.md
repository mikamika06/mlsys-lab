## Context

In transformer‑style models, the **PyramidKV** schedule allocates more key/value (KV) memory to lower layers and progressively less to higher layers while keeping the total budget fixed.  
Let $N$ be the number of layers and $B$ the total KV budget (an integer).  We wish to produce an allocation vector
$$\mathbf{a} = \bigl(a_0, a_1,\dots,a_{N-1}\bigr)$$
where $a_i$ is the number of KV slots kept for layer $i$.  
The desired properties are:

* **Monotonicity**: lower layers receive at least as many slots as higher ones,
  i.e. $a_{i} \ge a_{j}$ whenever $i > j$.
* **Budget conservation**: $\sum_{i=0}^{N-1} a_i = B$.

A simple deterministic rule that satisfies both is to assign weights proportional to the layer index (bottom layer gets the largest weight).  Let
$$w_i = i+1,\qquad i = 0,1,\dots,N-1,$$
and let $S=\sum_{i=0}^{N-1} w_i = \frac{N(N+1)}{2}$ be the total weight.  
The base allocation is then
$$\tilde a_i = \left\lfloor \frac{B\,w_i}{S}\right\rfloor,$$
which may leave a remainder $R=B-\sum_i \tilde a_i$.  The remaining slots are distributed one by one starting from the bottom layer (largest weight) until all $B$ slots are assigned.

This algorithm is fully deterministic, uses only integer arithmetic and NumPy vectorisation, and guarantees that the resulting allocation respects the pyramid property while exactly exhausting the budget.

## Task

Implement the function `pyramidkv_allocation`:

```python
import numpy as np

def pyramidkv_allocation(total_budget: int, num_layers: int) -> np.ndarray:
    ...
```

The function receives an integer total budget and the number of layers, and must return a one‑dimensional NumPy array of length `num_layers`.  Each element should be an integer (any signed dtype is acceptable).  The returned allocation must satisfy the properties described above.

## Example

```python
import numpy as np

# Total budget B = 7 over N = 4 layers
alloc = pyramidkv_allocation(7, 4)
print(alloc)          # [0 1 2 4]
```

Explanation:  
Weights are `[1, 2, 3, 4]`, sum `S=10`.  
Base allocation is `floor(7 * weights / 10)` → `[0, 1, 2, 3]` (sum 6).  
The remaining slot (`R = 1`) is given to the bottom layer, producing `[0, 1, 2, 4]`.

## What the gate checks

A single exact‑match gate verifies that the returned array equals a reference computed by an oracle implementation.  The grader recomputes the allocation using the same algorithm described above and compares it element‑wise with `np.array_equal`.  No hard‑coded expected values are used; the reference is generated on the fly for each test case.
