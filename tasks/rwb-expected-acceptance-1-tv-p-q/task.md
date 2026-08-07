## Context

In speculative decoding, a cheap draft model proposes a token from
distribution $q$, and the expensive target model's distribution $p$
decides whether to accept it. The standard accept/reject rule accepts a
proposed token $x \sim q$ with probability $\min\!\left(1, \frac{p(x)}{q(x)}\right)$.

Marginalizing over the draft's own choice of $x$, the **expected**
probability that a single proposed token gets accepted is

$$
\mathbb{E}[\text{accept}] = \sum_{x} q(x)\, \min\!\left(1, \frac{p(x)}{q(x)}\right)
= \sum_{x} \min\big(p(x), q(x)\big).
$$

This quantity has a second, equivalent closed form in terms of the
**total variation distance** between $p$ and $q$,
$\mathrm{TV}(p,q) = \frac12\sum_x |p(x)-q(x)|$:

$$
\sum_{x} \min(p(x), q(x)) = 1 - \mathrm{TV}(p,q) = 1 - \frac12\sum_x |p(x) - q(x)|.
$$

(This identity holds for any two probability distributions over the same
support: split the sum into where $p\ge q$ and where $p<q$; each half of
$\frac12\sum|p-q|$ exactly equals $\sum(\max(p,q)-\min(p,q))/2$, and
$\sum p = \sum q = 1$ pins the rest down — the two expressions for
acceptance probability aren't independent facts, they're the same number
written two ways.) So a draft distribution close to the target ($q\approx p$,
small TV) accepts almost always; a draft far from the target accepts
rarely.

## Task

Implement `expected_acceptance(p, q)`:

```python
def expected_acceptance(p: list[float], q: list[float]) -> float:
    ...
```

- `p`: list of floats, the target distribution (probabilities, sums to 1).
- `q`: list of floats, the draft distribution, same shape as `p` (sums to 1).

Return the expected single-token acceptance probability, computed via
**either** equivalent form above — $\sum_x \min(p_x,q_x)$ or
$1-\tfrac12\sum_x|p_x-q_x|$ — as a Python `float`.

## Example

```python

p = [0.7, 0.3]
q = [0.5, 0.5]

expected_acceptance(p, q)
# 0.8   (== sum(min([0.7,0.3],[0.5,0.5])) == 1 - 0.5*sum(|[0.2,-0.2]|))
```

Identical distributions accept with probability exactly `1.0`; totally
disjoint-support distributions accept with probability exactly `0.0`.

## What the gate checks

The oracle loads a fixed `(p, q)` pair from a fixture (softmax'd logits,
`q` a noisy perturbation of `p`) plus several additional seeded
`(p, q)` pairs of varying vocabulary size — including the identical-
distributions case (expected `1.0`) and a disjoint-support case (expected
`0.0`) — and for each one computes the reference **both** ways
($\sum\min$ and $1-\mathrm{TV}$), asserting internally that they agree
(they always must, by the identity above) before using that value as the
reference.

Your returned float is compared to the reference with the `rel_err`
scorer, and the worst case across every pair must be `< 1e-9`. A solution
that only implements one of the two forms correctly still passes (either
is accepted) — but averaging, clipping to `[0,1]` unnecessarily in a way
that changes an already-valid probability, or computing TV without the
$\tfrac12$ factor, will miss the tolerance.
