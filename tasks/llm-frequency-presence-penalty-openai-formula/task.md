## Context

Large language models often penalise the probability of tokens that have already appeared in a prompt. Two common penalties are **frequency** and **presence** penalties, introduced by OpenAI.  
For each token $t$ let  

* $c_t \in \mathbb{N}$ be the number of times $t$ has occurred in the prompt,
* $\alpha_{\text{freq}}$ be a non‑negative scalar controlling how much we penalise repeated tokens,
* $\alpha_{\text{pres}}$ be a non‑negative scalar controlling how much we penalise any token that appears at least once.

The OpenAI formula subtracts from each logit $l_t$ the amount

$$
\Delta_t = c_t \,\alpha_{\text{freq}} + \mathbf{1}_{c_t>0}\,\alpha_{\text{pres}},
$$

where $\mathbf{1}_{c_t>0}$ is 1 if $t$ has appeared and 0 otherwise.  
The adjusted logits are then

$$
l'_t = l_t - \Delta_t .
$$

This simple linear penalty encourages the model to generate novel tokens while still allowing frequent tokens when necessary.

## Task

Implement a function that applies this penalty to an array of logits.

```python
def apply_frequency_presence_penalty(
    logits: np.ndarray,
    token_counts: np.ndarray,
    freq_penalty: float,
    presence_penalty: float
) -> np.ndarray:
    ...
```

* `logits` is a 1‑D NumPy array of shape `(vocab_size,)` containing the raw logits for each token.
* `token_counts` is a 1‑D integer array of the same shape, giving $c_t$ for every token.
* `freq_penalty` and `presence_penalty` are floats $\alpha_{\text{freq}}$, $\alpha_{\text{pres}}$.

The function must return a new NumPy array of type `float64` containing the penalised logits.  No loops or explicit Python iteration may be used; rely on vectorised NumPy operations only.

## Example

```python
import numpy as np

logits = np.array([0.5, -1.2, 3.4], dtype=np.float64)
token_counts = np.array([2, 0, 1], dtype=np.int64)

# Frequency penalty 0.1, presence penalty 0.05
penalised = apply_frequency_presence_penalty(
    logits,
    token_counts,
    freq_penalty=0.1,
    presence_penalty=0.05
)
print(penalised)
```

Output:

```
[0.3   -1.2   3.25]
```

Explanation:  
Token 0 appears twice → penalty $2·0.1 + 0.05 = 0.25$;  
Token 1 never appears → penalty $0$;  
Token 2 appears once → penalty $1·0.1 + 0.05 = 0.15$.

## What the gate checks

The grader computes a reference implementation using NumPy and compares your output to it with the metric  

$$
\max_{t} \bigl|\, {l'}_t^{\text{your}} - {l'}_t^{\text{reference}} \bigr| .
$$

Your solution must achieve `max_abs_err <= 1e-6`.  Any deviation larger than this threshold will cause the gate to fail.  The grader also verifies that your function returns a `float64` array of the same shape as the input logits.
