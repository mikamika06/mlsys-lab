## Context

In transformer models, the self‑attention mechanism uses a query–key similarity matrix $QK^\top$ of shape $(N,N)$ for a sequence of length $N$. To enforce causality (each position can attend only to itself and earlier positions) we add an additive bias that is $-\infty$ on all entries above the main diagonal. The resulting masked logits are then passed through softmax.

Mathematically, if $L \in \mathbb{R}^{N\times N}$ denotes the raw logits, the causal mask $M$ is defined by

$$
M_{ij} = 
\begin{cases}
0 & i \ge j\\[4pt]
-\infty & i < j .
\end{cases}
$$

The masked logits are $L + M$.

## Task

Implement a function `causal_mask(logits)` that takes a 2‑D list of shape $(N,N)$ and returns the same shape with $-\infty$ added to all strictly upper‑triangular entries. The function must work for any numeric dtype; the output should be float64.

## Example

```python
logits = [[0, 1, 2],
                   [3, 4, 5],
                   [6, 7, 8]]
masked = causal_mask(logits)
print(masked)  # [[0.0, -inf, -inf], [3.0, 4.0, -inf], [6.0, 7.0, 8.0]]
```

## What the gate checks

The grader computes a reference mask using Python and compares your output with `max_abs_err`. The error must be at most $10^{-5}$. Additionally, the shape of the returned array must match the input.
