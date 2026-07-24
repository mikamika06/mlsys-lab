## Context

H2O-style attention compression keeps a subset of tokens by ranking them using an accumulated attention score. For an attention matrix $S \in \mathbb{R}^{n \times n}$, the attention probabilities are computed row by row:

$$
P_{ij} = \frac{\exp(S_{ij})}{\sum_{k=0}^{n-1}\exp(S_{ik})}.
$$

In autoregressive generation, attention is causal: token $i$ may only attend to tokens $j \le i$. Future tokens must be removed before computing probabilities. The causal mask is

$$
M_{ij} =
\begin{cases}
0, & j \le i \\
-\infty, & j > i .
\end{cases}
$$

The masked attention probabilities are therefore

$$
P_{ij} =
\mathrm{softmax}(S_{i,:}+M_{i,:})_j .
$$

The accumulated importance of token $j$ is the amount of attention it receives from all queries:

$$
h_j = \sum_{i=0}^{n-1} P_{ij}.
$$

H2O selects the tokens with the largest $h_j$ values. Applying the causal mask after this accumulation is incorrect because future attention changes the ranking.

## Task

Implement `select_heavy_hitters(attn_scores, budget)`:

```python
def select_heavy_hitters(attn_scores: np.ndarray, budget: int) -> np.ndarray:
    ...
```

The input is a square NumPy array of attention logits with shape $(n,n)$. Return a 1-D NumPy integer array containing the indices of the `budget` most important tokens according to causal H2O accumulation.

The implementation must:

1. Apply the causal mask before the softmax.
2. Accumulate the resulting attention probabilities by column.
3. Return token indices sorted by descending importance. Break ties by smaller token index.

The returned array must have dtype `int64`.

## Example

```python
import numpy as np

scores = np.array([
    [0.0, 8.0, 0.0],
    [0.0, 0.0, 8.0],
    [0.0, 0.0, 0.0],
])

kept = select_heavy_hitters(scores, 2)
# The future positions are masked before ranking.
# kept is a length-2 integer array containing the top causal heavy hitters.
```

## What the gate checks

The gate builds several attention score matrices and computes the expected retained set with a NumPy oracle that applies the causal mask before softmax accumulation.

The returned indices must exactly match the oracle. An implementation that accumulates the full non-causal attention matrix will produce a different ranking and fail.
