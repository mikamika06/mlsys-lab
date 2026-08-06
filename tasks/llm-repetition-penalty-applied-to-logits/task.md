## Context

In language modeling, a *repetition penalty* discourages the model from generating tokens that have already appeared in the current context.  
Given a vector of logits $\\mathbf{l} \\in \\mathbb{R}^{V}$ for a vocabulary of size $V$, and a set $S$ of token indices that have been generated so far, the penalty is applied as follows:

$$
l_i' = 
\begin{cases}
\dfrac{l_i}{p} & \text{if } i \\in S \text{ and } l_i > 0,\\[6pt]
l_i \cdot p   & \text{if } i \\in S \text{ and } l_i \\le 0,\\[6pt]
l_i           & \text{otherwise},
\end{cases}
$$

where $p>1$ is the penalty factor.  
This rule matches the implementation used in Hugging Face transformers.

## Task

Implement the function `apply_repetition_penalty` that applies the repetition penalty described above.

```python
def apply_repetition_penalty(logits: list[float],
                             seen_tokens: list[int],
                             penalty: float) -> list[float]:
    ...
```

* `logits` – a 1‑D list of shape `(V,)`, dtype `float64`.  
* `seen_tokens` – an iterable of integer token indices that have already appeared.  
* `penalty` – a positive scalar $p>1$.

The function must return a new list of the same shape and dtype as `logits`, with the penalty applied only to tokens in `seen_tokens`. Do **not** modify the input array in place.

## Example

```python

logits = [0.5, -1.2, 3.4, 0.0]
seen = [0, 2]
penalty = 2.0

new_logits = apply_repetition_penalty(logits, seen, penalty)
print(new_logits)  # [0.25, -1.2, 1.7, 0.0]
```

## What the gate checks

The grader computes a reference implementation using Python and compares your output with it.  
It reports the maximum absolute difference `max_abs_err`.  
Your solution must satisfy

$$
\mathrm{max\_abs\_err} \le 10^{-6}.
$$

Additionally, the returned array must have dtype `float64` and the same shape as the input logits.
