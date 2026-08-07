## Context

Split-KV / FlashAttention-2 style kernels compute attention over a long key
range by splitting the keys into shards, running an online-softmax pass
over each shard independently, and then **merging** the per-shard partial
results into the final output. Each shard $i$ produces a triple
$(m_i, l_i, o_i)$:

$$
m_i = \max_j \; \text{logits}_i[j], \qquad
l_i = \sum_j e^{\text{logits}_i[j] - m_i}, \qquad
o_i = \sum_j e^{\text{logits}_i[j] - m_i} \, v_j
$$

i.e. each shard's own numerator/denominator are computed relative to
**that shard's own local max** $m_i$, for numerical stability. Before the
shards can be combined, every partial has to be re-expressed on one common
scale — the **global** max $m = \max_i m_i$ — by multiplying it by
$e^{m_i - m}$:

$$
o = \frac{\sum_i e^{m_i - m}\, o_i}{\sum_i e^{m_i - m}\, l_i}
$$

Algebraically, merging with $e^{m_i}$ alone (instead of $e^{m_i - m}$)
gives the *same* mathematical answer — the $m_i$ cancels out of numerator
and denominator either way. But $e^{m_i}$ on its own is **not** bounded by
1, so for realistic logit magnitudes it silently overflows to `inf`, and
`inf / inf` becomes `nan`. Subtracting the global max is not optional
cosmetics; it is the only thing standing between this merge and a
guaranteed overflow once the logits are more than a few hundred in
magnitude — exactly the range real (unscaled) attention logits reach.

## Task

Fix `merge_split_kv(partials)`:

```python
def merge_split_kv(partials: list[tuple[float, float, list[float]]]) -> list[float]:
    ...
```

`partials` is a list of `(m_i, l_i, o_i)` triples as defined above (`m_i`,
`l_i` floats, `o_i` a 1-D array — one triple per KV shard, all shards
covering the same query). The provided implementation combines them by
scaling each partial by `exp(m_i)` and never computes the global max `m`
across shards at all. Fix the rescale so the merge matches the exact
full-sequence attention output, including on inputs where the local maxes
`m_i` are large.

## Example

```python
# two shards, logits large enough that m_i is in the hundreds
partials = [merge_shard1, merge_shard2]  # each (m_i, l_i, o_i)
merge_split_kv(partials)
# buggy version: exp(m_i) overflows -> inf, inf/inf -> nan
# fixed version: exp(m_i - max(m_i for all shards)) stays <= 1 -> exact,
#   finite output equal to running full-precision softmax attention over
#   every key across all shards at once
```

## What the gate checks

The gate builds a handful of split-KV scenarios from a seeded generator:
random queries/keys/values scaled so that per-shard local max logits land
in the tens-to-thousands range (as real, not-yet-`1/sqrt(d)`-tamed
attention logits can), split into 3-4 uneven shards each. For every case
it computes the reference the honest way — running full fp64 softmax
attention over the *entire* concatenated key/value sequence at once (no
sharding) — and compares it to your merged output.

Your output must be **finite** and within `max_abs_err < 1e-5` of that
reference on every case. The provided (buggy) merge is mathematically
equivalent in exact arithmetic but always produces `nan` at this logit
scale, because it scales by `exp(m_i)` instead of `exp(m_i - m)` and
overflows before the division ever happens — so it fails on every case,
for a real numerical reason, not a contrived one.
