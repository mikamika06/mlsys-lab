## Context

Speculative decoding uses a smaller draft model to propose several tokens and a larger model to verify them. For a draft block of tokens $x_1,\dots,x_k$, the verifier compares the draft distribution $q_i$ and target distribution $p_i$ at each position.

A token at position $i$ is accepted with probability

$$
\min\left(1, \frac{p_i(x_i)}{q_i(x_i)}\right).
$$

Verification proceeds left-to-right. The first rejected position ends the accepted prefix. After a rejection, a bonus token is sampled from the residual distribution

$$
r_i(j) = \frac{\max(p_i(j)-q_i(j),0)}
{\sum_l \max(p_i(l)-q_i(l),0)} .
$$

If every draft token is accepted, the bonus token is sampled from the final target distribution $p_k$ instead. A deterministic random stream is provided so that sampling can be reproduced.

## Task

Implement `verify_block(draft, p, q, rng)`:

```python
def verify_block(draft, p, q, rng):
    ...
```

Arguments:

- `draft` is a list of integer token ids of length $k$.
- `p` is a list of $k$ probability lists. `p[i][j]` is the target probability of token $j$ at position $i$.
- `q` is a list of $k$ probability lists with the same shape as `p`.
- `rng` is a list of floating point values in $[0,1)$ used in order.

Return a tuple:

```python
(accepted_length, emitted_tokens)
```

where `accepted_length` is the number of consecutive draft tokens accepted and `emitted_tokens` contains the accepted prefix followed by exactly one sampled bonus token.

For sampling, consume the next random value after the acceptance checks. Given a probability distribution, choose the smallest token id whose cumulative probability is strictly greater than the random value.

## Example

```python
draft = [2, 1]
p = [
    [0.1, 0.2, 0.7],
    [0.5, 0.4, 0.1],
]
q = [
    [0.2, 0.3, 0.5],
    [0.4, 0.5, 0.1],
]
rng = [0.1, 0.8, 0.2]

result = verify_block(draft, p, q, rng)
# deterministic output: accepted_length and emitted_tokens
```

## What the gate checks

The gate computes the same block verification algorithm independently and compares the returned tuple exactly on several draft blocks, probability tables, and random streams.

A solution passes only if it accepts the longest valid prefix and samples the correct bonus token from the correct distribution.
