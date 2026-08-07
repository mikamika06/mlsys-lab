## Context

Softmax is used to convert logits $z \in \mathbb{R}^d$ into a probability distribution  
$$p_i = \frac{\exp(z_i)}{\sum_{j=1}^{d}\exp(z_j)}.$$  
When computing attention scores we often need to mask out certain positions (e.g. padding tokens or future positions). The correct way is to set the logits of masked positions to $-\infty$ before applying softmax, so that their exponentials become zero and the remaining probabilities are renormalised automatically.

## Task

Implement `masked_softmax(logits, mask)`:

```python
def masked_softmax(logits: list[list[float]], mask: list[list[bool]]) -> list[list[float]]:
    ...
```

`logits` is a 2‑D array of shape $(B, L)$ containing raw attention scores.  
`mask` is a boolean array of the same shape; `True` indicates that the corresponding position must be masked out.

The function should return an array of shape $(B, L)$ with probabilities that sum to one along the last axis for every batch element.

## Example

```python
logits = [[1.0, 2.0, -1.0],
                   [0.5, 0.0, 3.0]]
mask   = [[False, True, False],
                   [True,  False, False]]

probs = masked_softmax(logits, mask)
print(probs)  # [[0.8807970779778823, 0.0, 0.11920292202211755], [0.0, 0.04742587317756679, 0.9525741268224334]]
```

## What the gate checks

The relative L2 error between your output and a reference implementation must satisfy  
$$\mathrm{rel\_err} \le 10^{-6}.$$
