## Context

In multi‑head self‑attention each head operates on a pair of matrices \(K,V \in \mathbb{R}^{n\times d}\).  
The output is
\[
O = \operatorname{softmax}\!\left(\frac{K K^\top}{\sqrt{d}}\right) V .
\]
When deploying models on hardware with limited precision, we often quantise \(K\) and \(V\) to a small number of bits. Because the two matrices play different roles—\(K\) determines the attention weights while \(V\) contributes to the final values—it is not obvious which one should receive more bits for a fixed total budget.

## Task

Implement `classify_high_bits(K, V, total_bits=8)`:

```python
def classify_high_bits(K: list[list[float]], V: list[list[float]], total_bits: int=8) -> int:
    ...
```

The function must return `0` if allocating the higher precision to \(K\) (and lower to \(V\)) yields a smaller mean‑squared error in the attention output than the opposite allocation; otherwise it should return `1`.  
Use only Python operations; no explicit Python loops.

## Example

```python
K = [[0.0, 1.0], [2.0, -1.0]]
V = [[1.0, 0.5], [-0.5, 2.0]]
idx = classify_high_bits(K, V, total_bits=8)
print(idx)   # e.g., 0
```

## What the gate checks

The grader evaluates both possible bit allocations with a fixed split of `total_bits-1` bits for the high‑precision side and `1` bit for the low‑precision side.  
It computes the attention output in full precision, then quantises \(K\) and \(V\) according to each allocation, recomputes the output, and measures the mean‑squared error against the reference.  
The assignment that yields the lower error is considered correct; the function must return its index (`0` for “\(K\) high”, `1` for “\(V\) high”). Any deviation causes the gate to fail.
