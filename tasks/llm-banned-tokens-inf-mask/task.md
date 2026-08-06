## Context

In language‑model decoding, it is common to forbid certain token indices from being sampled.
The standard way to do this is to set the corresponding logits to $-\infty$, which makes the softmax probability for that token exactly zero:

$$p_i = \frac{e^{\ell_i}}{\sum_j e^{\ell_j}}, \qquad \text{if }\ell_k = -\infty \;\Rightarrow\; p_k = 0.$$

This operation is often called *masking* or *banning* tokens.

## Task

Implement the function `mask_banned_tokens` that takes a list of logits and a list of banned token indices, and returns a new list where every entry whose column index is in the banned list has been replaced by $-\infty$.
The function must work for both 1‑D lists (a single set of logits) and 2‑D lists (a batch of logits). It must **not** modify its input.

```python
def mask_banned_tokens(logits: list[float] | list[list[float]], banned_indices: list[int]) -> list[float] | list[list[float]]:
    ...
```

## Example

```python

# Single row of logits
logits = [0.5, -1.2, 3.4]
banned = [1, 2]
masked = mask_banned_tokens(logits, banned)
print(masked)          # [ 0.5 -inf -inf]

# Batch of logits
batch = [[0.5, -1.2, 3.4],
                  [1.0, 0.0, -0.5]]
masked_batch = mask_banned_tokens(batch, banned)
print(masked_batch)
# [[ 0.5 -inf -inf]
#  [ 1.0 -inf -inf]]
```

## What the gate checks

The grader computes a reference list using Python and compares it to the candidate’s output with an exact‑match check (`==`).

A correct implementation must return a list that is element‑wise identical, including the $-\infty$ values. No other metrics are evaluated.
