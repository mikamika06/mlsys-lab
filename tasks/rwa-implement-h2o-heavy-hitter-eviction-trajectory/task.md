## Context

H2O (Heavy-Hitter Oracle) keeps a KV cache under a fixed `budget` by
tracking, for every resident token, the **cumulative attention mass**
it has ever received, and evicting the lowest-scoring token whenever the
cache would otherwise grow past budget — except for a protected
`recent_window` of the most-recently-appended tokens, which can never be
evicted (a token needs a chance to accumulate score before it's judged).

At decode step $t$ (0-indexed), with resident positions $\mathcal{R}_t$:

$$
w_t = \operatorname{softmax}\!\Big(\tfrac{q_t\, K_{\mathcal R_t}^\top}{\sqrt d}\Big),
\qquad
\text{score}_i \mathrel{+}= (w_t)_i \ \ \forall i \in \mathcal{R}_t
$$

Then the newly generated position `prompt_len + t` is appended to the
cache with `score = 0`. If $|\mathcal{R}_t| + 1 > \text{budget}$, evict
the single lowest-scoring position among those **not** in the
`recent_window` most-recently-appended resident positions (ties broken
by smaller position index).

## Task

Implement `h2o_eviction_trajectory`:

```python
def h2o_eviction_trajectory(
    K: np.ndarray, Q: np.ndarray,
    prompt_len: int, budget: int, recent_window: int,
) -> list[list[int]]:
    ...
```

- `K`: `(prompt_len + T, d)` keys for the prompt and every token that
  will be decoded, indexed by absolute position.
- `Q`: `(T, d)`, the query issued at each decode step.
- `prompt_len`: number of prompt positions initially resident
  (`0..prompt_len-1`); guaranteed `<= budget`.
- `budget`: maximum resident cache size.
- `recent_window`: number of most-recently-appended resident positions
  that are always protected from eviction (`>= 1`).
- For each of the `T` decode steps, in order: attend `Q[t]` over the
  currently resident positions (ascending order) to accumulate scores,
  append position `prompt_len + t` with score `0`, then evict if over
  budget as described above.
- Return a list of length `T`; entry `t` is the **sorted** list of
  resident position indices immediately after step `t`.

## Example

```python
import numpy as np

rng = np.random.default_rng(0)
prompt_len, T, d = 3, 5, 4
K = rng.standard_normal((prompt_len + T, d))
Q = rng.standard_normal((T, d))

traj = h2o_eviction_trajectory(K, Q, prompt_len=3, budget=4, recent_window=1)
# len(traj) == 5; each entry is a sorted list of <= 4 resident positions.
# traj[0] has 4 entries (3 prompt + 1 new, no eviction needed yet).
# traj[1] onward: one eviction happens each step to stay at budget 4.
```

## What the gate checks

The grader builds several seeded `(K, Q, prompt_len, budget,
recent_window)` decode simulations and replays the exact same recurrence
independently in NumPy — accumulate softmax attention mass into a score
dict, append the new position, evict the arg-min-score non-protected
resident when over budget — never calling your function, never
hardcoding an expected trajectory.

`exact_match` is the fraction of decode steps, across all cases, whose
resident set both (a) has size `<= budget` and (b) exactly equals the
oracle's resident set at that step. The gate requires `1.0`. Scoring
against the wrong query, protecting the wrong window, evicting by
largest instead of smallest score, or mis-handling a tie all cause the
resident set to diverge from the oracle at the very next step (and stay
diverged, since eviction state carries forward).
