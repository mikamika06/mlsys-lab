## Context

The softmax function maps a vector of logits $z \in \mathbb{R}^n$ to a probability distribution:

$$\operatorname{softmax}(z)_i = \frac{\exp(z_i)}{\sum_{j=1}^{n}\exp(z_j)}.$$

Numerically it is common to subtract the maximum logit before exponentiating, because this does not change the result but prevents overflow:

$$\operatorname{softmax}(z)_i
   = \frac{\exp(z_i - M)}{\sum_{j=1}^{n}\exp(z_j - M)}, \qquad
  M = \max_k z_k.$$

In many high‑performance libraries a *block‑wise* or *streaming* softmax is used.  
The idea is to process the logits in blocks of size $B$, compute a local maximum for each block, exponentiate with that local shift, and then rescale so that the final result equals the monolithic softmax.

Let $z$ be split into consecutive blocks $b_1,\dots,b_K$.  
For block $k$ let

$$m_k = \max_{i\in b_k} z_i.$$

Define a scaling factor for each block

$$s_k = \exp(m_k - M),$$

where $M=\max_i z_i$ is the global maximum.  
Then for any index $i$ in block $k$ we have

$$
\frac{\exp(z_i)}{\sum_j \exp(z_j)}
= \frac{\exp(z_i-m_k)\,s_k}{\sum_{j}\exp(z_j-M)}.
$$

Thus a correct block‑wise implementation must:

1. Compute the global maximum $M$ once.
2. For each block compute its local maximum $m_k$, exponentiate with that shift, multiply by $s_k$, and accumulate the denominator.
3. Finally divide every scaled exponential by the accumulated denominator.

## Task

Implement the function

```python
def block_rescale_softmax(logits: list[float], block_size: int) -> list[float]:
    ...
```

It takes a 1‑D list of logits and an integer `block_size`.  
Return a 1‑D array of softmax probabilities with dtype `float64`.  
The implementation must use only Python operations; the only Python loop allowed is to iterate over blocks.

## Example

```python
logits = [0.2, -1.5, 3.0, 0.7]
D = block_rescale_softmax(logits, block_size=2)
# D ≈ [0.0144, 0.0006, 0.9838, 0.0012]
```

The result matches `[math.exp(x - max(logits)) / sum(math.exp(x - max(logits)) for x in logits) for x in logits]` to machine precision.

## What the gate checks

The grader computes a reference softmax using Python and compares it with the candidate’s output for several random test cases and block sizes.  
It reports the maximum absolute error:

$$\mathrm{err} = \max_i |\,\hat p_i - p_i\,|.$$

The gate requires $\mathrm{err}\le 10^{-6}$.
