## Context

Collective communication primitives are the backbone of parallel computing.  
The *all‑gather* operation collects data from every process and distributes the
concatenated result back to all processes. In a ring implementation each rank
sends its local payload to the next rank in a circular topology, receiving a
payload from the previous rank, until all ranks have seen every piece of data.

Mathematically, if there are $N$ ranks and each rank $i$ holds an array
$\mathbf{x}_i \in \mathbb{R}^{m}$, then after a correct all‑gather every rank
holds

$$\mathbf{y} = \bigl[\mathbf{x}_0^\top,\;\mathbf{x}_1^\top,\;\dots,
\mathbf{x}_{N-1}^\top\bigr]^\top \in \mathbb{R}^{Nm}\,.$$

The task is to implement a pure‑Python simulation of this operation.

## Task

Implement the function `ring_all_gather` that accepts a list of NumPy arrays
representing the local data on each rank and returns a list of NumPy arrays,
one per rank, containing the fully gathered buffer as described above.

```python
def ring_all_gather(local_arrays: list[np.ndarray]) -> list[np.ndarray]:
    ...
```

The function must:

1. Accept any number of ranks $N \ge 1$.
2. Preserve the order of data from rank 0 to rank $N-1$ in the output buffer.
3. Return a new array for each rank; modifying one should not affect the
   others.

## Example

```python
import numpy as np

local_arrays = [
    np.array([10, 20]),
    np.array([30, 40]),
    np.array([50, 60])
]

gathered = ring_all_gather(local_arrays)

# Each element of `gathered` is the same concatenated array:
print(gathered[0])  # [10 20 30 40 50 60]
print(gathered[1])  # [10 20 30 40 50 60]
print(gathered[2])  # [10 20 30 40 50 60]
```

## What the gate checks

The grader computes a reference buffer by concatenating all local arrays.
For each rank it evaluates the maximum absolute difference between the
candidate’s output and the reference. The candidate passes if

$$\max_{i}\;\bigl\lVert \text{output}_i - \text{reference} \bigr\rVert_\infty
   \le 10^{-6}\,.$$

Only correctness is checked; algorithmic complexity is not penalised.
