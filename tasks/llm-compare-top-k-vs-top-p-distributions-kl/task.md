## Context

Language models produce a probability distribution over a vocabulary before sampling a token. Sampling methods modify this distribution before choosing an output token.

For logits $z \in \mathbb{R}^V$, the softmax distribution is

$$
P_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}} .
$$

Top-k sampling keeps only the $k$ highest-probability tokens and renormalizes them. Top-p sampling keeps the smallest set of highest-probability tokens whose cumulative probability reaches a threshold $p$, then renormalizes the retained probabilities.

The difference between two sampling filters can be measured with KL divergence:

$$
D_{KL}(P \| Q) = \sum_i P_i \log \frac{P_i}{Q_i}.
$$

This task compares the renormalized top-k and top-p distributions and computes the divergence between them.

## Task

Implement `top_k_top_p_kl(logits, k, p)`.

The function receives:

- `logits`: a list of lists of floats of shape $(n, V)$ containing model logits.
- `k`: the number of tokens retained by top-k sampling.
- `p`: the cumulative probability threshold for nucleus (top-p) sampling.

Return a Python `float` equal to the mean KL divergence across all rows between the renormalized top-k distribution and the renormalized top-p distribution:

$$
\frac{1}{n}\sum_{r=1}^{n} D_{KL}(K_r \| P_r).
$$

Use Python operations. The implementation should compute the softmax probabilities, apply both filtering strategies independently, renormalize each filtered distribution, and then compute the divergence.

For top-p, tokens must be selected after sorting probabilities in descending order. Include the first token that makes the cumulative probability reach or exceed $p$.

## Example

```python

logits = [[3.0, 2.0, 1.0, 0.0]]
value = top_k_top_p_kl(logits, k=2, p=0.8)
```

The function compares the top-k distribution over the two highest-probability tokens with the top-p distribution over the smallest prefix of sorted probabilities reaching $0.8$.

## What the gate checks

The gate recomputes the expected KL divergence using a Python reference implementation of softmax, top-k filtering, top-p filtering, and KL divergence. The returned value must match the oracle result within numerical tolerance.
