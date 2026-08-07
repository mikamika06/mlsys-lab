## Context

Mixture-of-Experts (MoE) layers route tokens to a set of experts. In token-choice
routing, each token chooses its highest-scoring expert. This can overload popular
experts because many tokens may select the same destination.

Expert-choice routing reverses the direction of selection. Each expert chooses
its best tokens. Given a routing score matrix $S \in \mathbb{R}^{T \times E}$,
where $T$ is the number of tokens and $E$ is the number of experts, expert $e$
selects the tokens with the largest values in column $S_{:,e}$.

If the capacity factor is $c$, each expert receives

$$
C = \left\lceil c \frac{T}{E} \right\rceil
$$

token slots. A token can appear in multiple expert assignments, while a token
that is never selected by any expert is dropped.

For comparison, token-choice routing assigns

$$
a_i = \arg\max_e S_{i,e}
$$

for each token $i$. Tokens assigned beyond expert capacity are dropped.

## Task

Implement `expert_choice_routing(logits, capacity_factor)`:

```python
def expert_choice_routing(logits: list[list[float]], capacity_factor: float):
    ...
```

The input `logits` is a list of lists of floats of shape $(T, E)$. Return a tuple:

```python
(expert_tokens, expert_dropped, token_choice_dropped)
```

where:

- `expert_tokens` is a list of length $E$. Each element is a list of token
  indices selected by that expert in descending score order. Equal scores are
  resolved by smaller token index first.
- `expert_dropped` is the number of tokens that are not selected by any expert.
- `token_choice_dropped` is the number of dropped tokens under token-choice
  routing with the same capacity $C$. Token-choice assigns each token to its
  highest-scoring expert, then keeps only the first $C$ tokens per expert using
  the same score ordering rule.

Use Python operations for score processing. The returned lists and integers must
match the reference behavior exactly.

## Example

```python

logits = [
    [3.0, 1.0],
    [2.0, 4.0],
    [0.5, 2.5],
]

expert_tokens, expert_dropped, token_choice_dropped = expert_choice_routing(
    logits, 1.0
)

# capacity = ceil(1.0 * 3 / 2) = 2
# expert_tokens == [[0, 1], [1, 2]]
# expert_dropped == 0
# token_choice_dropped == 1
```

## What the gate checks

The gate builds a Python reference implementation of both routing strategies and
compares the complete returned tuple. The comparison checks the selected token
lists, expert-choice dropped-token count, and token-choice dropped-token count.

A solution that only implements token-choice routing fails because expert-choice
allows experts to independently select their highest scoring tokens.
