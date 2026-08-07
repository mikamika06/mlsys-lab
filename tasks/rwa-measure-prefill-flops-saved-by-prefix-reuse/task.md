## Context

Prefix caching (vLLM's automatic prefix caching, SGLang's RadixAttention)
lets a new request skip recomputing the prefill forward pass for any
leading tokens it shares with a previously-cached sequence — the cached
KV entries are reused directly instead of being recomputed.

Model the causal prefill cost of a single request of length $L$ under
the standard assumption that processing the token at position $i$ costs
FLOPs proportional to $i$ (it must attend over all $i$ prior tokens):

$$
\text{cost}(L) = \sum_{i=1}^{L} i = \frac{L(L+1)}{2}.
$$

If the first $p$ tokens of the request are already cached from a prior
request that shares that prefix, only positions $p+1, \dots, L$ need to
actually run through the model — but each of those tokens **still**
attends across the full context (both the newly computed keys and the
reused, cached ones), so it still costs its own index $i$ in FLOPs:

$$
\text{cost}_{\text{reuse}}(L, p) = \sum_{i=p+1}^{L} i = \frac{L(L+1)}{2} - \frac{p(p+1)}{2}.
$$

## Task

Implement `prefill_flops_saved_fraction`:

```python
def prefill_flops_saved_fraction(lengths: list[float], reused_prefix: list[float]) -> float:
    ...
```

* `lengths` — shape $(n,)$, the full context length $L_i$ of each of $n$
  requests in a batch.
* `reused_prefix` — shape $(n,)$, the number of leading tokens of
  request $i$ already cached from a shared prefix ($0 \le
  \text{reused\_prefix}_i \le L_i$).

For each request compute $\text{cost}(L_i)$ and
$\text{cost}_{\text{reuse}}(L_i, p_i)$ as above, then return the
fraction of **total batch** prefill FLOPs saved by reuse:

$$
\text{saved\_fraction} = 1 - \frac{\sum_i \text{cost}_{\text{reuse}}(L_i, p_i)}{\sum_i \text{cost}(L_i)} .
$$

## Example

```python

lengths = [100, 100]
reused_prefix = [0, 100]   # first request: no reuse; second: fully cached

saved = prefill_flops_saved_fraction(lengths, reused_prefix)
# request 1 saves nothing, request 2 saves everything, so saved is
# exactly the second request's share of total full-prefill cost: 0.5
# here since both have the same length.
```

## What the gate checks

The gate, **rel_err**, compares your returned fraction against an
analytic oracle across several random batches (including a no-reuse
batch, where the saved fraction must be exactly 0, and a fully-reused
batch, where it must be exactly 1). Your result must match to a relative
error of `<= 1e-6`.
