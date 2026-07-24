## Context

Speculative decoding uses a cheap **draft** model with distribution $q$ to
propose a token, and a **target** model with distribution $p$ to verify it,
without ever losing the guarantee that the final sampled token is
distributed exactly as $p$. The accept/reject/resample scheme (Leviathan et
al., Chen et al.) works like this for one token position:

1. Sample a draft token $x \sim q$.
2. **Accept** $x$ with probability $\min\!\big(1, \tfrac{p(x)}{q(x)}\big)$.
3. If **rejected**, sample a replacement token $y$ from the *residual*
   distribution
   $$
   p_{\text{res}}(y) \;=\; \frac{\max(p(y) - q(y),\, 0)}{\sum_{z} \max(p(z) - q(z),\, 0)} .
   $$

The residual formula is the non-obvious part, and it is what makes the
scheme exact. Since $p(y) = \min(p(y), q(y)) + \max(p(y) - q(y), 0)$ for
every $y$, summing the "accept" and "reject-then-resample" branches gives
$$
\underbrace{\min(p(x), q(x))}_{\text{accept branch}} \;+\;
\underbrace{P(\text{reject}) \cdot p_{\text{res}}(x)}_{\text{resample branch}}
\;=\; p(x),
$$
so the marginal distribution of the *final* output token is exactly $p$ —
no bias, even though only $q$ was ever sampled from directly for the draft.

It's tempting — and wrong — to think "if we rejected, just sample the
replacement straight from the target distribution $p$", since $p$ is what
we ultimately want. That skips the correction for probability mass the
draft already contributed via the accept branch, and it biases the output:
the mass at each token gets counted once from the (nonzero) accept branch
*and again* from an uncorrected $p$-resample, instead of splitting exactly
$p(y) = \min(p,q)(y) + \max(p-q,0)(y)$ between the two branches.

## Task

Implement the residual (rejection) distribution:

```python
def residual_distribution(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    ...
```

* `p` — 1-D array of shape $(V,)$, the target model's distribution over a
  vocabulary of size $V$ (`p.sum() == 1`, `p >= 0`).
* `q` — 1-D array of shape $(V,)$, the draft model's distribution over the
  same vocabulary (`q.sum() == 1`, `q >= 0`).

Return the array $p_{\text{res}}$ of shape $(V,)$: elementwise
$\max(p - q, 0)$, renormalized to sum to 1. This is the distribution that
should be sampled from whenever the accept/reject step above rejects the
draft token — **not** $p$ itself.

## Example

```python
import numpy as np
p = np.array([0.1, 0.6, 0.3])
q = np.array([0.5, 0.4, 0.1])

residual_distribution(p, q)
# max(p-q,0) = [0, 0.2, 0.2], sum = 0.4
# -> array([0. , 0.5, 0.5])
```

## What the gate checks

A single gate, **mean_kl**, runs the *entire* accept/reject/resample loop
end-to-end (draft sampling from $q$, accept-probability test against $p$,
and — on rejection — resampling from **your** `residual_distribution(p, q)`)
for 200,000 trials, on two fixed, meaningfully-different `(p, q)` pairs. It
tallies the empirical output-token frequency for each pair and compares it
to the true target $p$ with $\mathrm{KL}(p \,\|\, \text{empirical})$. The
mean of the two pairs' KL values must be $\le 5\times 10^{-3}$. A residual
distribution equal to $p$ (the bug) biases the empirical output distribution
enough to fail this by roughly an order of magnitude; the correct
$\max(p-q,0)$-renormalized residual keeps the whole scheme unbiased, so with
200,000 samples the empirical KL is driven down by sampling noise alone,
comfortably under the threshold. All randomness inside the gate is
seeded (`np.random.default_rng(0)`), so the result is deterministic.
