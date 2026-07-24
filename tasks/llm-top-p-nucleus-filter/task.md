## Context

Language models produce a probability distribution over the next-token vocabulary. Sampling from the full distribution can include many low-probability tokens, so nucleus sampling keeps only a small set of likely tokens.

Given token probabilities $p_1, p_2, \dots, p_n$, top-$p$ sampling first sorts tokens by probability in descending order:

$$
p_{(1)} \ge p_{(2)} \ge \dots \ge p_{(n)} .
$$

The nucleus is the smallest prefix of this ordering whose cumulative probability reaches the threshold $p$:

$$
\sum_{i=1}^{k-1} p_{(i)} < p
\quad\text{and}\quad
\sum_{i=1}^{k} p_{(i)} \ge p .
$$

The boundary token at position $k$ must be included. The filter returns the original token indices in the nucleus.

## Task

Implement `top_p_filter(probs, p)`:

```python
def top_p_filter(probs: np.ndarray, p: float) -> np.ndarray:
    ...
```

The input `probs` is a one-dimensional NumPy array of non-negative probabilities whose values sum to approximately $1$. The input `p` is a threshold in $(0,1]$.

Return a one-dimensional NumPy array containing the original indices of the tokens kept by top-$p$ nucleus sampling. Indices may be returned in any order, but every selected index must belong to the smallest sorted probability prefix whose cumulative probability is at least $p$.

Use NumPy operations for sorting and cumulative sums.

## Example

```python
import numpy as np

probs = np.array([0.05, 0.60, 0.10, 0.25])
kept = top_p_filter(probs, 0.70)

# Sorted probabilities are:
# token 1: 0.60, token 3: 0.25, token 2: 0.10, token 0: 0.05
# The first two tokens reach 0.85, so the nucleus is {1, 3}.
```

## What the gate checks

The gate computes the expected nucleus using a NumPy reference implementation that sorts the probabilities, computes the cumulative sum, and includes the first token where the cumulative probability reaches the threshold.

The returned index set must exactly match the reference nucleus on several probability distributions and thresholds. The metric is `exact_match`, which must equal $1.0$.
