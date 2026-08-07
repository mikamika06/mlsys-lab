## Context

Chunked and ring-attention implementations split the key/value sequence
into chunks processed independently (by different pipeline stages or
different devices). For one query row, each chunk $c$ computes its own
raw scores $\{s_{c,i}\}$ and reduces them to a single scalar, its
**log-sum-exp**:

$$
\mathrm{LSE}_c = \log \sum_i e^{s_{c,i}} .
$$

This is the quantity FlashAttention-family kernels actually keep around
(for the backward pass, or to exchange between distributed workers)
instead of the raw $O(N)$ score vector. The key identity that makes
chunking exact is that the *global* log-sum-exp over every chunk combined
is itself just the log-sum-exp of the per-chunk log-sum-exps:

$$
\sum_{c} e^{\mathrm{LSE}_c} = \sum_c \sum_i e^{s_{c,i}} = \sum_{\text{all } j} e^{s_j}
\quad\Longrightarrow\quad
\mathrm{LSE}_{\text{global}} = \log \sum_c e^{\mathrm{LSE}_c}.
$$

Once you have $\mathrm{LSE}_{\text{global}}$, the true global softmax
weight of **any individual token** $j$ (in any chunk) is recovered
directly from its own raw score:

$$
w_j = \frac{e^{s_j}}{\sum_{\text{all}} e^{s}} = e^{\,s_j - \mathrm{LSE}_{\text{global}}}.
$$

This is exactly how a coordinator (or a later kernel stage) can recover
every token's true global attention weight having only ever seen $C$
scalar LSE reductions plus each chunk's own local scores — never a
materialized $O(N)$ combined score array.

## Task

Implement `reconstruct_global_weights(chunk_scores, chunk_lse, chunk_partial_out)`:

```python
def reconstruct_global_weights(chunk_scores: list[list[float]], chunk_lse: list[float], chunk_partial_out: list[list[float]]) -> list[float]:
    ...
```

- `chunk_scores`: shape `(C, chunk_size)` — each token's raw score,
  grouped by chunk.
- `chunk_lse`: shape `(C,)` — each chunk's own log-sum-exp, i.e.
  `chunk_lse[c] == logsumexp(chunk_scores[c])`.
- `chunk_partial_out`: shape `(C, d)` — each chunk's local unnormalized
  partial output (given for context, as a real chunked-attention worker
  would have it on hand; not required to compute the weights).

Compute $\mathrm{LSE}_{\text{global}} = \log\sum_c e^{\mathrm{LSE}_c}$
(numerically stably — subtract $\max_c \mathrm{LSE}_c$ before
exponentiating), then return the flattened `(C * chunk_size,)` vector of
$w_j = e^{s_j - \mathrm{LSE}_{\text{global}}}$, in the same
`(chunk, position-within-chunk)` order as `chunk_scores.reshape(-1)`.

## Example

```python

chunk_scores = [[1.0, 2.0], [0.5, 3.0]]   # C=2, chunk_size=2
chunk_lse = [math.log(math.exp(1.0) + math.exp(2.0)),
math.log(math.exp(0.5) + math.exp(3.0))]
chunk_partial_out = [[0.0] * 4 for _ in range(2)]  # unused here

w = reconstruct_global_weights(chunk_scores, chunk_lse, chunk_partial_out)
# w.shape == (4,), w.sum() == 1.0, and w matches a plain softmax over
# [1.0, 2.0, 0.5, 3.0] exactly.
```

## What the gate checks

The gate loads a fixed 6-chunk, 4-tokens-per-chunk fixture (raw scores
with a wide dynamic range, plus the corresponding real per-chunk LSEs and
partial outputs, all derived from the same underlying scores by
construction), plus several seeded synthetic cases with different chunk
counts/sizes. For each case, the oracle computes the reference weight
vector **independently** — a single plain softmax over every score
concatenated together, entirely bypassing the per-chunk LSE machinery, so
it can't accidentally share a bug with any particular reconstruction
method.

Your returned vector is compared to the oracle's with the `rel_err`
scorer (global relative L2 error), and the worst case across every
scenario must be `< 1e-6`. Any correct combination method passes — the
two-level log-sum-exp shown above is the intended one, but the gate only
checks the resulting numbers. Getting the numerically-unstable version
wrong on the wide-dynamic-range fixture (e.g. `logsumexp(chunk_lse)`
without subtracting its own max before exponentiating) or mixing up
chunk/within-chunk ordering when flattening will miss the tolerance.
