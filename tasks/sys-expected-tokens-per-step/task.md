## Context

In draft-target speculative decoding, a small draft model proposes $K$ tokens
per step, and the large target model verifies them in a single forward pass.
Verification is sequential: position $i$ is accepted with probability $p_i$
*given that every earlier position was accepted*, and the first rejection
stops the chain — everything after it is discarded.

Whether the chain stops early (rejection) or all $K$ draft tokens survive, the
target model always emits exactly one more token that step: either the
corrected token that replaces the rejected draft token, or, if all $K$ were
accepted, a fresh bonus token sampled from the target's own distribution. So
if $L$ is the (random) number of accepted draft tokens, the number of tokens
produced that step is always $L + 1$.

Let $L$ be the number of accepted draft tokens before the first rejection
(with $L = K$ if all are accepted). Its distribution is

$$
\Pr[L = \ell] =
\begin{cases}
\left(\prod_{i=1}^{\ell} p_i\right)(1 - p_{\ell+1}) & 0 \le \ell < K \\[4pt]
\prod_{i=1}^{K} p_i & \ell = K
\end{cases}
$$

and the expected number of tokens emitted in one speculative decoding step is

$$
E[\text{tokens per step}] = \sum_{\ell=0}^{K} (\ell + 1)\,\Pr[L = \ell] .
$$

This telescopes into the closed form

$$
E[\text{tokens per step}] = 1 + \sum_{i=1}^{K} \prod_{j=1}^{i} p_j .
$$

## Task

Implement `expected_tokens_per_step(accept_probs)`.

`accept_probs` is a 1D sequence (list or list) of $K$ floats in
$[0, 1]$, where `accept_probs[i]` is the acceptance probability of draft
position $i$ conditional on all earlier positions being accepted. $K$ may be
`0` (an empty draft — the target still emits exactly one token that step).

Return the expected number of tokens emitted per speculative decoding step as
a Python `float`, using the closed form above.

## Example

```python
accept_probs = [0.5, 0.5]

# E = 1 + 0.5 + 0.5*0.5 = 1.75
result = expected_tokens_per_step(accept_probs)
```

## What the gate checks

The gate computes the expected value independently by summing over the exact
stopping-position distribution $\Pr[L=\ell]$ for $\ell = 0, \dots, K$ (rather
than the telescoped closed form), on several random and edge-case probability
vectors (including all-zero, all-one, and empty). It compares your result
against this reference with

$$
\mathrm{rel\_err} = \frac{|x - \hat{x}|}{\max(|x|, 10^{-15})}
$$

which must satisfy $\mathrm{rel\_err} \le 10^{-9}$. Forgetting the guaranteed
"+1" bonus token, or using unconditional (non-cumulative) acceptance
probabilities, both fail this gate.
