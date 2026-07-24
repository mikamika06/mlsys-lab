## Context

Top-p (nucleus) sampling keeps the smallest set of tokens whose cumulative probability
mass reaches a threshold $p$.

Given probabilities $q_1, q_2, \dots, q_n$, first sort tokens by probability:

$$q_{(1)} \ge q_{(2)} \ge \dots \ge q_{(n)}.$$

The cumulative probability after keeping the first $k$ tokens is

$$C_k = \sum_{i=1}^{k} q_{(i)}.$$

A boundary detail matters when $C_k$ is exactly equal to the threshold. The correct
rule includes the token that reaches the boundary:

$$C_k \ge p.$$

A buggy implementation often uses $C_k > p$, which removes a token when the
probability mass lands exactly on the threshold.

## Task

Implement `top_p_keep(probs, p)`:

```python
def top_p_keep(probs: list[float], p: float) -> list[int]:
    ...
```

The function receives a list of token probabilities and a top-p threshold. It must
return the original indices of the tokens kept by top-p sampling.

Requirements:

- Sort tokens by probability in descending order.
- Include tokens until the first cumulative probability satisfies
  $C_k \ge p$.
- Return indices in descending probability order.
- Do not return sorted probabilities; return original token indices.

## Example

```python
probs = [0.40, 0.35, 0.25, 0.10]

top_p_keep(probs, 0.75)
# [0, 1]
```

The first two tokens have cumulative probability $0.75$, so the second token is
included.

## What the gate checks

The gate computes the expected kept token indices using a NumPy reference
implementation of the cumulative probability boundary rule. The returned list must
match the oracle exactly on several cases, including probabilities where the
cumulative sum is exactly equal to $p$.
