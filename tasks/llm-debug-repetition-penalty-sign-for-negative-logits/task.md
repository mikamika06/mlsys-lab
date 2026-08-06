## Context

Language models produce a vector of logits $z \in \mathbb{R}^V$, where each entry
$z_i$ is the score of token $i$ before converting scores into probabilities.

A repetition penalty modifies logits for tokens that have already appeared. A
common rule with penalty $p > 1$ is:

$$
z_i' =
\begin{cases}
z_i / p & \text{if } z_i > 0, \\
z_i \cdot p & \text{if } z_i < 0, \\
0 & \text{if } z_i = 0.
\end{cases}
$$

The sign matters because positive logits are reduced while negative logits are
pushed farther from zero. Applying the same division operation to every value
incorrectly makes negative logits less negative and changes the token ranking.

## Task

Implement `apply_repetition_penalty(logits, penalty)`:

```python
def apply_repetition_penalty(logits: list[float], penalty: float) -> list[float]:
    ...
```

The function receives a list of floats of logits and a scalar
penalty. Return a new `float64` list where every element is transformed
according to the sign-aware rule above.

Do not modify the input array.

## Example

```python

logits = [4.0, -3.0, 0.0, 1.5]
out = apply_repetition_penalty(logits, 2.0)

# [2.0, -6.0, 0.0, 0.75]
```

## What the gate checks

The gate computes the expected output using a Python reference implementation of
the sign-aware repetition penalty rule. The returned array must have
maximum absolute error

$$
\max_i |y_i - \hat{y}_i| < 10^{-6}.
$$

Implementations that divide all logits by the penalty fail because negative
values require multiplication by the penalty.
