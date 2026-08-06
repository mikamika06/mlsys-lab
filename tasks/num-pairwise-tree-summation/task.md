## Context

When adding many floating‑point numbers, the order of operations can influence the final result because each addition introduces a rounding error. A classic remedy is *Kahan summation*, which keeps an auxiliary compensation term. Another powerful technique is to perform the additions in a balanced binary tree: split the array into two halves, sum each half recursively, and finally add the two partial sums. This divide‑and‑conquer strategy reduces the depth of the accumulation chain from $O(n)$ to $O(\log n)$, thereby limiting error growth.

## Task

Implement `pairwise_sum`:

```python
def pairwise_sum(arr: list[float]) -> float:
    ...
```

The function receives a list of floats of type `float64` and must return the sum of all elements using a tree‑based algorithm. The implementation should be fully vectorised where possible, but recursion is allowed for splitting the data. Empty arrays should yield `0.0`.

## Example

```python
arr = [1.0, 2.0, 3.0, 4.0]
s = pairwise_sum(arr)
print(s)          # 10.0
```

## What the gate checks

The grader computes a reference sum with the built-in `sum` function and compares it to your result using the relative error metric:

$$\mathrm{rel\_err} = \frac{|\,\text{your\_sum} - \text{ref\_sum}\,|}{|\text{ref\_sum}| + 10^{-12}}.$$

The gate requires $\mathrm{rel\_err} \le 1\times10^{-9}$.
