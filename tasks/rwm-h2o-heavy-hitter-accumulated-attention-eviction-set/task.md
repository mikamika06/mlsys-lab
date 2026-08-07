## Context

H2O (Heavy-Hitter Oracle) compresses a KV cache by keeping only the tokens
that matter most. For a causal attention logit matrix $S \in
\mathbb{R}^{n\times n}$, the masked, row-wise softmax gives the attention
probabilities

$$
P_{ij} = \operatorname{softmax}\!\big(S_{i,:} + M_{i,:}\big)_j,
\qquad
M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}.
$$

Every token $j$'s **accumulated importance** is the total attention mass it
has ever received, summed down its column:

$$
h_j = \sum_{i=0}^{n-1} P_{ij}.
$$

H2O keeps a fixed `budget` of tokens made of two groups:

1. the **recent window** — the `recent_window` most-recently-appended
   positions (indices $n-\text{recent\_window}, \dots, n-1$), always kept
   regardless of score (a token needs time to accumulate attention before
   being judged), and
2. the **heavy hitters** — the `budget - recent_window` tokens with the
   largest $h_j$ among all positions *outside* the recent window.

The union of these two groups is the retained set. Its **preserved mass**
is the fraction of total accumulated importance it captures:

$$
\text{preserved\_mass} = \frac{\sum_{j \in \text{retained}} h_j}{\sum_{j=0}^{n-1} h_j}.
$$

## Task

Implement `h2o_eviction_set`:

```python
def h2o_eviction_set(attn_scores: list[list[float]], budget: int, recent_window: int):
    ...
```

* `attn_scores` — `(n, n)` list of raw (pre-softmax) attention
  logits.
* `budget` — number of tokens to keep; guaranteed `recent_window <= budget <= n`.
* `recent_window` — number of most-recent positions always kept;
  guaranteed `1 <= recent_window <= budget`.

Steps:

1. Apply the causal mask, then softmax row-wise, to get $P$.
2. Accumulate importance per token as the column sums $h_j$.
3. Take the recent window (the last `recent_window` indices) as always
   kept.
4. From the remaining candidates, take the `budget - recent_window` with
   the largest $h_j$ (break ties by the smaller index).
5. Return `(retained_idx, preserved_mass)`:
   - `retained_idx`: 1-D `int64` list of the retained indices, in
     **ascending sorted order**, length exactly `budget`.
   - `preserved_mass`: a plain Python `float`, as defined above.

## Example

```python

rng = random.Random(0)
S = rng.standard_normal((8, 8))
S[:, 1] += 15.0  # token 1 becomes a clear heavy hitter

retained_idx, preserved_mass = h2o_eviction_set(S, budget=4, recent_window=2)
# retained_idx always contains {6, 7} (the recent window for n=8),
# plus the 2 highest-h_j indices among {0,1,2,3,4,5} — token 1 among them.
# preserved_mass is close to 1.0 since heavy hitters + recent capture
# most of the accumulated attention.
```

## What the gate checks

The grader builds several seeded `(attn_scores, budget, recent_window)`
cases with Python — including a case with one deliberately dominant column
to sanity-check the ranking direction — and independently computes the
causal-masked softmax, column-summed importance, heavy-hitter selection,
and preserved mass, exactly as described above (never calling your
function, never hardcoding an expected answer).

`exact_match` requires, on every case: your `retained_idx` array is
**exactly** equal (same indices, same ascending order) to the oracle's, and
your `preserved_mass` matches the oracle's to within `1e-9`. Forgetting the
causal mask, accumulating over the wrong axis, protecting the wrong window,
selecting the lowest instead of the highest scores, or mis-handling ties
will all produce a different retained set and fail the gate.
