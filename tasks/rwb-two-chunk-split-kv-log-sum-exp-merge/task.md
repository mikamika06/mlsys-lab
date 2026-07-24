## Context

During decoding, a single query token can attend over a very long KV
cache. "Split-KV" (a.k.a. flash-decoding) kernels parallelize this single
query's attention by splitting the KV cache into chunks, computing each
chunk's attention **independently**, and then combining the per-chunk
results — without ever materializing the full score row across the whole
cache at once.

For one query $q$ and a KV chunk with keys $K^{(i)}$, values $V^{(i)}$,
the scaled scores are $s^{(i)}_j = q^\top K^{(i)}_j / \sqrt{d}$. Chunk $i$
computes its own local, normalized softmax output

$$
O_i = \frac{\sum_j \exp(s^{(i)}_j) V^{(i)}_j}{\sum_j \exp(s^{(i)}_j)}
$$

(computed stably with a local max-shift), along with its chunk's true
log-sum-exp value

$$
L_i = \log \sum_j \exp\!\big(s^{(i)}_j\big).
$$

Because each chunk's $O_i$ was normalized using only *its own* local
softmax denominator, the chunks' outputs cannot simply be averaged — a
chunk whose scores happen to be larger contributed disproportionately to
the true, whole-cache softmax and must be weighted more. The $L_i$ values
are exactly the right per-chunk weight (in log-space) to correct for
this. For two chunks, the merged output is

$$
m = \max(L_1, L_2), \qquad
O = \frac{O_1\, e^{L_1 - m} + O_2\, e^{L_2 - m}}{e^{L_1 - m} + e^{L_2 - m}} ,
$$

which reproduces the exact single-pass attention output over the whole
(unsplit) KV cache — this is the log-sum-exp correction every split-KV /
flash-decoding kernel relies on.

## Task

Implement `two_chunk_split_kv_merge(q, k, v)`:

```python
def two_chunk_split_kv_merge(q, k, v):
    ...
```

- `q`: NumPy array of shape $(d,)$ — a single decode query.
- `k`, `v`: NumPy arrays of shape $(N, d)$ and $(N, d_v)$ — the full KV
  cache for this query.

1. Split `k`/`v` into exactly two contiguous chunks at the midpoint:
   chunk 1 is rows `[0, N//2)`, chunk 2 is rows `[N//2, N)`.
2. For each chunk, compute its local normalized output $O_i$ and its true
   log-sum-exp $L_i$ from that chunk's scaled scores
   ($1/\sqrt{d}$ scale), as defined above.
3. Merge the two chunks' results with the log-sum-exp correction formula
   above.

Return the merged output as a `float64` NumPy array of shape $(d_v,)$.

## Example

A tiny 1-D sanity check: if chunk 1's scores are all far larger than
chunk 2's, $L_1 \gg L_2$, so $e^{L_2-m} \approx 0$ and the merged output
is essentially just $O_1$ — chunk 2 barely influences the result, exactly
as it would if a single softmax over the whole cache were computed
directly (chunk 2's tiny scores would get almost no softmax weight there
either).

## What the gate checks

The gate loads the committed `q.npy`/`k.npy`/`v.npy` fixture (one query,
a 200-row KV cache) and computes a full, single-pass `float64`
scaled-dot-product-attention oracle directly over the *entire* (unsplit)
KV cache. Your two-chunk merged output is compared against this oracle
with `max_abs_err`, threshold $10^{-5}$. Averaging the two chunks' `O_i`
directly (ignoring $L_i$), using the raw unnormalized per-chunk sums
instead of $O_i$, or forgetting the $m = \max(L_1, L_2)$ shift will not
reproduce the true single-pass result.
