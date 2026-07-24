## Context

In language modeling, a model outputs logits for each token in the vocabulary. Sometimes we need to restrict the set of tokens that can be chosen at a particular decoding step—for example, when following a grammar or applying a mask. A common technique is to set the logits of disallowed tokens to $-\infty$ before selecting the next token. This guarantees that the argmax will never pick a forbidden token.

## Task

Implement `masked_greedy(logits, allowed_sets)`:

```python
def masked_greedy(logits: np.ndarray,
                  allowed_sets: Iterable[Iterable[int]]) -> np.ndarray:
    ...
```

- `logits` is a 2‑D NumPy array of shape $(n,\;v)$ where $n$ is the number of decoding steps and $v$ is the vocabulary size.
- `allowed_sets[i]` contains the indices of tokens that are permitted at step $i$. Each set is non‑empty.
- The function must return a 1‑D array of length $n$ containing, for each step, the index of the token with the largest logit after disallowing all other tokens.

The implementation should use only NumPy; no explicit Python loops over the steps are required but may be used if desired. The output dtype should be `np.int64`.

## Example

```python
import numpy as np

logits = np.array([[0.1, 2.3, -1.5],
                   [4.0, 0.2, 1.1]])
allowed_sets = [{0, 2}, {1}]
chosen = masked_greedy(logits, allowed_sets)
print(chosen)   # [2, 1]
```

In the first step token 2 has the largest logit among the allowed tokens {0, 2}. In the second step only token 1 is allowed.

## What the gate checks

The grader generates several random test cases. For each case it computes a reference solution by masking disallowed logits with `-np.inf` and taking `argmax`. The candidate’s output must match this reference exactly for all steps; otherwise the gate fails.
