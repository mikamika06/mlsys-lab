## Context

Speculative decoding's verification step uses **modified rejection
sampling** to decide, position by position, whether to keep a drafted
token or replace it — in a way that makes the final output distribution
exactly match the target model, never the draft model.

At position $t$, let $p_t$ and $q_t$ be the target and draft probability
distributions over the vocabulary, and let $x_t$ be the token the draft
model proposed there. The token is **accepted** with probability

$$
\alpha(x_t) = \min\!\left(1, \frac{p_t(x_t)}{q_t(x_t)}\right),
$$

by drawing a fresh uniform $u \sim \mathrm{Unif}(0,1)$ and accepting iff
$u \le \alpha(x_t)$. If accepted, the emitted token is $x_t$ itself. If
**rejected**, a replacement token is drawn instead from the *residual*
distribution

$$
r_t(x) = \frac{\max(p_t(x) - q_t(x),\ 0)}{\displaystyle\sum_{x'} \max(p_t(x') - q_t(x'),\ 0)},
$$

using **the next** uniform draw from the same stream (inverse-CDF
sampling: the smallest index $i$ with $\sum_{x \le i} r_t(x) \ge u$).
Crucially, an accepted position consumes only **one** uniform draw; a
rejected position consumes **two** (the accept-check draw, then the
resample draw) — so the stream position advances by a variable amount
depending on what happened at each prior step, not a fixed 1 or 2 per
position.

## Task

Implement `modified_rejection_sample(p, q, draft_token_ids, u_stream)`:

```python
def modified_rejection_sample(p: list[list[float]], q: list[list[float]], draft_token_ids: list[int], u_stream: list[float]) -> list[int]:
    ...
```

- `p`, `q`: list of shape $(T, V)$ — target and draft
  distributions for each of $T$ positions, each row summing to $1$.
- `draft_token_ids`: integer array of shape $(T,)$, the drafted token id
  at each position.
- `u_stream`: 1-D float array of pre-drawn uniforms in $[0,1)$, long
  enough to cover the worst case of every position being rejected (length
  $2T$). Consume it **sequentially from the front**, advancing a single
  shared pointer across positions in order: one draw for the accept
  check at every position, plus one more only when that position is
  rejected.

For every position $t = 0, \dots, T-1$, in order:
1. Pop the next value from `u_stream` as `u_accept`.
2. Accept iff `u_accept <= min(1, p[t, draft_token_ids[t]] / q[t, draft_token_ids[t]])`
   (treat the ratio as `0` if `q[t, draft_token_ids[t]] == 0`). If
   accepted, the emitted token for position $t$ is `draft_token_ids[t]`.
3. Otherwise, pop the next value from `u_stream` as `u_resample`, build
   $r_t$ from `p[t]` and `q[t]` as above, and emit the residual sample:
   the smallest index whose cumulative sum of $r_t$ is `>= u_resample`
(`bisect.bisect_left([sum(r_t[:i+1]) for i in range(len(r_t))], u_resample)`, clipped
   to a valid index).

Return an integer list of shape $(T,)$: the emitted token id at
each position.

## Example

For a 2-token vocabulary at one position, `p = [0.4, 0.6]`,
`q = [0.7, 0.3]`, drafted token `x = 0`:
`p[0]/q[0] = 0.571...`, so acceptance probability is `0.571...`. If
`u_accept = 0.8 > 0.571...`, the draft is rejected. The residual is
`r = normalize([max(0.4-0.7,0), max(0.6-0.3,0)]) = normalize([0, 0.3]) =
[0, 1]` — token `1` is emitted for any `u_resample > 0`.

## What the gate checks

The gate loads a committed `p.npy`/`q.npy`/`draft_token_ids.npy`/
`u_stream.npy` fixture (25 positions, 12-token vocabulary) and replays
the same sequential accept/resample simulation independently, using the
exact same stream-pointer discipline described above. Your returned
per-position emitted token ids are compared against the oracle's for
exact equality at every position (`exact_match == 1.0`). Consuming a
fixed number of draws per position (e.g. always 2, or always 1)
desynchronizes the shared pointer after the first rejection or
acceptance and causes every subsequent position to diverge from the
oracle.
