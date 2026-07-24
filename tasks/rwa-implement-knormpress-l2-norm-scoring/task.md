## Context

In many production systems a dictionary of integer keys maps to high‑dimensional feature vectors. When memory is limited we often keep only the *top‑\(k\)* entries whose vectors have the largest Euclidean norm, discarding the rest. This compression technique is called **KNormPress**.

For a vector \(v \in \mathbb{R}^d\) its L2 norm is

$$
\lVert v\rVert_2 = \sqrt{\sum_{i=1}^{d} v_i^2}\,.
$$

The KNormPress algorithm selects the keys with the largest norms up to a given budget \(k\). If fewer than \(k\) keys exist all are kept. The output is a list of the selected integer keys sorted in ascending order.

## Task

Implement the function

```python
def knormpress(data: dict[int, np.ndarray], budget: int) -> list[int]:
    ...
```

* `data` maps integer keys to NumPy arrays (1‑D vectors).  
* `budget` is a non‑negative integer.  
* Return a **list of the selected keys** sorted in ascending order.

The implementation must be fully deterministic and use only NumPy for numeric work; no Python loops over the dictionary items are required but allowed if you wish.

## Example

```python
import numpy as np

data = {
    10: np.array([3.0, 4.0]),      # norm = 5
    2 : np.array([1.0, 1.0]),      # norm ≈ 1.414
    7 : np.array([0.0, 0.0]),      # norm = 0
    5 : np.array([6.0, 8.0])       # norm = 10
}

# Keep the two largest norms (budget=2)
kept = knormpress(data, 2)
print(kept)   # [5, 10]
```

## What the gate checks

The grader computes a reference solution using NumPy and compares your output **exactly**.  
If the returned list differs in any element or order, the `exact_match` metric fails.

No other metrics are evaluated; the focus is on correctness of the selection logic.
