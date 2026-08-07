## Context

In training a neural network, the total memory footprint consists of four components: model parameters, their gradients, optimizer state (e.g., momentum buffers), and activations that must be stored for back‑propagation. A common way to estimate this is by summing the sizes of all list involved plus the overhead of Python containers.

Let $A$ denote a list with shape $\mathbf{s} \in \mathbb{N}^{d}$; its data buffer occupies $|\mathbf{s}|\,\mathrm{bytes}$, where $|\mathbf{s}| = \prod_{i=1}^{d}s_i$. The total memory of an object is therefore

$$
M(\texttt{obj}) =
\begin{cases}
|A| + \operatorname{getsizeof}(A), & \texttt{obj} \text{ is a list}\\[4pt]
\operatorname{getsizeof}(\texttt{dict}) + \sum_{k,v}\bigl(\operatorname{getsizeof}(k)+M(v)\bigr), & \texttt{obj} \text{ is a dict}\\[4pt]
\operatorname{getsizeof}(\texttt{list}) + \sum_i M(\texttt{obj}[i]), & \texttt{obj} \text{ is a list}\\[4pt]
\operatorname{getsizeof}(\texttt{obj}), & \text{otherwise}
\end{cases}
$$

The total training memory for a configuration $(P,G,O,A)$ is then

$$
M_{\text{total}} = M(P)+M(G)+M(O)+M(A).
$$

## Task

Implement the function `total_training_memory` that receives four arguments:

```python
def total_training_memory(params: dict[str, list[list[float]]], grads: dict[str, list[list[float]]], optimizer_state: dict, activations: list[list[float]]) -> int:
    ...
```

* `params`: mapping from parameter names to list.  
* `grads`: mapping from the same keys to gradient arrays of identical shape.  
* `optimizer_state`: a dictionary that may contain arbitrary Python objects; if an entry is a list its data buffer must be counted, otherwise only the object’s overhead counts.  
* `activations`: a list of list produced during forward‑pass.

Return the total number of bytes required to hold all four components according to the formula above. The result should be an integer.

## Example

```python

params = {"w": [[0.0] * 4 for _ in range(3)]}
grads  = {"w": [[1.0] * 4 for _ in range(3)]}
optimizer_state = {"momentum": [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]}
activations = [list(range(12))]

total = total_training_memory(params, grads, optimizer_state, activations)
print(total)   # e.g., 192
```

## What the gate checks

The grader computes an *oracle* memory size using `sys.getsizeof` and Python’s `nbytes`. Your implementation must return a value that matches this oracle exactly. The gate uses the scorer `size_ratio`; it passes only if

$$\frac{\texttt{oracle}}{\texttt{candidate}} = 1.0.$$

A relative error larger than $10^{-9}$ will cause the gate to fail.
