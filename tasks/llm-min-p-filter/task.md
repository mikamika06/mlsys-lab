## Context

In language‑model sampling, a common strategy is to filter out low‑probability tokens before selecting one at random. One such technique is the **min‑p filter**: given a probability distribution over tokens $p_i$ and a threshold parameter $\alpha \in (0,1]$, we keep only those tokens whose probability exceeds a fraction of the maximum probability:
$$\text{keep}(i) = \begin{cases} \text{True} & \text{if } p_i \ge \alpha \cdot \max_j p_j,\\ \text{False} & \text{otherwise.} \end{cases}$$
This simple rule can dramatically reduce the candidate set while preserving tokens that are relatively likely.

## Task

Implement a function `minp_filter` that receives:

* `probs`: a 1‑D list of token probabilities (dtype float64).
* `min_p`: a scalar in $(0,1]$ representing $\alpha$.

The function must return a boolean mask of the same shape as `probs`, where each element is `True` if and only if its probability satisfies the min‑p condition above. The implementation should be fully vectorized using Python; no explicit Python loops are allowed.

```python
def minp_filter(probs: list[float], min_p: float) -> list[bool]:
    ...
```

## Example

```python
probs = [0.05, 0.15, 0.30, 0.50]
mask = minp_filter(probs, 0.5)
print(mask)
# [False False  True  True]
```

Here $\max p_i = 0.50$, the threshold is $0.5 \times 0.50 = 0.25$, and tokens with probabilities $0.30$ and $0.50$ are kept.

## What the gate checks

The grader computes a reference mask using the exact formula above and compares it to your output. The metric **exact_match** must be `1.0` for the solution to pass; any discrepancy yields `0.0`.
