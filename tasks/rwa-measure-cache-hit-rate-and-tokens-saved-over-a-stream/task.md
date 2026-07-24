## Context

A request cache can avoid recomputing work when the same request appears again.
For a stream of tokenized requests, a cache hit occurs when the current request
has already appeared earlier in the stream.

If request $i$ contains $t_i$ tokens, the total number of reused tokens is the
sum of token counts for requests that were served from cache:

$$
R = \sum_{i \in H} t_i ,
$$

where $H$ is the set of stream positions that are cache hits.

The cache hit rate measures how often requests can be reused:

$$
\mathrm{hit\_rate} = \frac{|H|}{N},
$$

where $N$ is the total number of requests.

A simple exact matcher can determine whether a request was seen before by
checking all previous requests. Production systems replace this with hash-based
structures, but the measured result should match the exact reuse semantics.

## Task

Implement `measure_cache_stats(requests)`:

```python
def measure_cache_stats(requests):
    ...
```

`requests` is a list of requests. Each request is a list of integer token IDs.

Return a tuple:

```python
(hit_rate, reused_tokens)
```

where:

- `hit_rate` is a `float` equal to the fraction of requests that were cache hits.
- `reused_tokens` is an `int` equal to the total number of tokens in requests that
  were cache hits.

A request is a hit only if the complete sequence of token IDs has appeared
earlier in the stream. The order of tokens matters.

## Example

```python
requests = [
    [10, 11, 12],
    [5, 6],
    [10, 11, 12],
    [5, 6, 7],
    [5, 6],
]

hit_rate, reused_tokens = measure_cache_stats(requests)

# hit_rate == 0.4
# reused_tokens == 5
```

The third request reuses $3$ tokens and the fifth request reuses $2$ tokens.

## What the gate checks

The gate replays each request stream with an exact reference matcher and
compares the returned values against the oracle result.

The relative error

$$
\mathrm{rel\_err} =
\frac{\lVert y_{\mathrm{student}} - y_{\mathrm{oracle}} \rVert_2}
{\lVert y_{\mathrm{oracle}} \rVert_2 + 10^{-12}}
$$

must satisfy $\mathrm{rel\_err} \le 10^{-6}$.

The reference checks both the hit rate and the reused-token count.
