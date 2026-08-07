## Context

Speculative decoding proposes $\gamma$ tokens with a cheap **draft**
model, then checks all of them in a single **target** model forward
pass. If each proposed token is accepted independently with probability
$\alpha$ (the acceptance rate), the number of tokens actually emitted in
one round — the accepted prefix plus one guaranteed "bonus" token from
the target's own distribution at the first rejection (or after all
$\gamma$ are accepted) — has expectation

$$
\mathbb{E}[\text{tokens per round}] = \sum_{k=0}^{\gamma} \alpha^{k}
= \frac{1-\alpha^{\gamma+1}}{1-\alpha}.
$$

Let $c$ be the cost of one draft forward pass relative to one target
verification pass ($c = \text{cost}_{\text{draft}} /
\text{cost}_{\text{target}}$, typically $c \ll 1$ since the draft model
is much smaller). One round costs $\gamma c + 1$ target-equivalent
units ($\gamma$ draft steps plus one target verification pass, which
checks all $\gamma{+}1$ positions in parallel for the price of one
pass). The expected **speedup** over plain autoregressive decoding with
the target alone (which emits exactly 1 token per target-equivalent
unit of cost) is

$$
\text{speedup}(\alpha,\gamma,c) =
\frac{\mathbb{E}[\text{tokens per round}]}{\gamma c + 1}
= \frac{1}{\gamma c + 1}\sum_{k=0}^{\gamma}\alpha^{k}.
$$

## Task

Implement `speculative_speedup(alpha, gamma, cost_ratio)`:

```python
def speculative_speedup(alpha: float, gamma: int, cost_ratio: float) -> float:
    ...
```

- `alpha`: per-token draft-acceptance probability, `0 <= alpha <= 1`.
- `gamma`: number of draft tokens proposed per round (positive int).
- `cost_ratio`: $c$ as defined above.

Return the analytic expected speedup from the formula above. Use the
geometric **sum** form $\sum_{k=0}^{\gamma}\alpha^k$ rather than the
closed division form — the division form is $0/0$ exactly at
$\alpha=1$ even though the true limit is $\gamma+1$.

## Example

```python
speculative_speedup(alpha=0.8, gamma=4, cost_ratio=0.2)
# E[tokens] = 1 + 0.8 + 0.64 + 0.512 + 0.4096 = 3.3616
# cost = 4*0.2 + 1 = 1.8
# speedup = 3.3616 / 1.8 ≈ 1.8676

speculative_speedup(alpha=1.0, gamma=4, cost_ratio=0.2)
# every draft token is always accepted -> E[tokens] = gamma + 1 = 5
# speedup = 5 / 1.8 ≈ 2.7778
```

## What the gate checks

The grader evaluates 13 cases — 10 random `(alpha, gamma, cost_ratio)`
triples (randomly generated) plus `alpha=0`, `alpha=1`
exactly, and `gamma=1` — against the same formula computed independently
in `check.py`. The gate metric `size_ratio` is
`min(got/expected, expected/got)` (1.0 = exact match), required
`>= 1 - 1e-9`. Using the closed-form division instead of the geometric
sum will diverge or NaN at `alpha == 1`; forgetting the `+1` for the
target verification pass, or normalizing by `gamma` instead of by the
per-round cost, will be off by a large, easily-caught margin on nearly
every case.
