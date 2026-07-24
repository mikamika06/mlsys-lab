## Context

In language‑model decoding we often transform raw logits before turning them into a probability distribution. Two common transformations are **temperature scaling** and various **penalties** (repetition penalty, logit bias, banned‑token mask). The order in which these operations are applied can change the final distribution.

Temperature scaling multiplies all logits by $1/T$ where $T>0$. A temperature $T<1$ sharpens the distribution, while $T>1$ smooths it. Penalties typically subtract or add a constant to selected logits; for example a repetition penalty might reduce the logit of a token that has already appeared.

The Kullback–Leibler divergence between two categorical distributions $p$ and $q$ is

$$
\mathrm{KL}(p \,\|\, q) = \sum_{i} p_i \log \frac{p_i}{q_i}.
$$

We will compute the **mean** KL over a batch of logits.

## Task

Implement `compare_penalty_temperature(logits, penalty_fn, temperature)`:

```python
def compare_penalty_temperature(
    logits: np.ndarray,
    penalty_fn: Callable[[np.ndarray], np.ndarray],
    temperature: float
) -> float:
    ...
```

* `logits` is a 2‑D NumPy array of shape `(batch, vocab)` containing raw model logits.
* `penalty_fn` takes an array of logits and returns the penalised logits (e.g. subtracting a constant or masking certain tokens).
* `temperature` is a positive float.

The function must:

1. Compute two probability distributions:
   * **Before‑temp**: apply `penalty_fn` first, then divide by `temperature`, finally softmax.
   * **After‑temp**: divide logits by `temperature` first, then apply `penalty_fn`, finally softmax.
2. Return the mean Kullback–Leibler divergence of the two distributions over the batch.

The implementation must use only NumPy (no explicit Python loops) and produce a `float64` result.

## Example

```python
import numpy as np

def bias_penalty(x):
    return x - 0.5   # simple constant bias

logits = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
temp = 1.5

kl = compare_penalty_temperature(logits, bias_penalty, temp)
print(kl)   # e.g. 0.0123456789
```

## What the gate checks

The grader computes a reference value using the same algorithm and compares it to your output with a relative error tolerance of $10^{-9}$. Your implementation must match that reference exactly.
