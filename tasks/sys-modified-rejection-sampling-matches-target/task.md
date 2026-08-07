## Context

Speculative decoding proposes a block of $K$ tokens with a cheap draft
model and verifies them against an expensive target model. At each
verified position $k$, the draft proposed a token from distribution
$q_k$ over the vocabulary, and the target's distribution at that
position is $p_k$. **Modified rejection sampling** turns $(p_k, q_k)$
and a draft sample $x \sim q_k$ into a sample that is, provably,
distributed exactly as $p_k$ — never worse than the target model, no
matter how bad the draft is:

$$
\text{accept } x \text{ with probability } \quad \alpha_k(x) = \min\!\left(1,\ \frac{p_k(x)}{q_k(x)}\right).
$$

If $x$ is rejected, resample a *fresh* token from the **residual
distribution**

$$
r_k(v) = \frac{\max(p_k(v) - q_k(v),\ 0)}{\displaystyle\sum_{v'} \max(p_k(v') - q_k(v'),\ 0)} .
$$

The output token's marginal distribution — across many independent
repetitions of "draw $x\sim q_k$, accept/reject, resample from $r_k$ on
reject" — is exactly $p_k$, for *any* draft distribution $q_k$ with the
same support considerations. This is the correctness property that makes
speculative decoding lossless.

## Task

Implement `rejection_sample_block`:

```python
def rejection_sample_block(P: list[list[float]], Q: list[list[float]], n_draws: int, seed: int) -> list[list[float]]:
    ...
```

* `P`, `Q` — `float64` arrays of shape $(K, V)$: row $k$ is the target
  distribution $p_k$ and draft distribution $q_k$ (each row sums to 1)
  over a vocabulary of size $V$, for verification position $k$.
* `n_draws` — number of independent repetitions of the accept/resample
  procedure to run **per position** $k$.
* `seed` — integer seed for `random.Random(seed)`, used to drive
  your random draws.

For every position $k = 0,\dots,K-1$ independently: repeat `n_draws`
times the procedure "draw $x \sim q_k$, accept with probability
$\alpha_k(x)=\min(1, p_k(x)/q_k(x))$, otherwise resample from $r_k$",
and record the resulting output token each time.

Return an array `E` of shape $(K, V)$ where `E[k]` is the **empirical
distribution** (normalized token counts) of the `n_draws` output tokens
at position $k$. For large `n_draws`, `E[k]` should closely approximate
`P[k]`.

## Example

```python

P = [[0.70, 0.20, 0.10]]
Q = [[0.40, 0.35, 0.25]]

E = rejection_sample_block(P, Q, n_draws=200_000, seed=0)
print(E)   # ~= [[0.70, 0.20, 0.10]]
```

## What the gate checks

The grader builds several deterministic `(P, Q, n_draws, seed)` cases and
runs its own reference implementation of the exact same procedure
(independently seeded) to get a target frequency table. **mean_kl** is
the average, over all rows of all cases, of
$\mathrm{KL}(P_k \,\Vert\, E_k)$ in nats, using your returned empirical
distribution `E_k` in place of the true target `P_k`. It must be
$\le 10^{-2}$ — achievable only if your accept probability and residual
distribution are implemented exactly as defined above (a scheme that
resamples from `P` directly on rejection, or omits the `min(1, ·)` clip,
or forgets to renormalize the residual, biases `E_k` away from `P_k` by
far more than this tolerance, even with many draws).
