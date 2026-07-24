## Context

Grouped-query attention (GQA) reduces key/value cache bandwidth by sharing key/value heads between multiple query heads. Let $n_q$ be the number of query heads and let the group factor $g$ mean that each key/value head serves $g$ query heads. The number of key/value heads is therefore

$$
n_{kv} = \frac{n_q}{g}.
$$

For one autoregressive decode step, consider the attention score computation and value aggregation. The dominant floating point work is modeled as two matrix multiplications per query head:

$$
\mathrm{FLOPs} = 4 n_q L d,
$$

where $L$ is the cached sequence length and $d$ is the head dimension.

Assume the key and value cache are stored in fp16. The cache traffic is

$$
\mathrm{KV\ bytes} = 4 L d \frac{n_q}{g}.
$$

The query input traffic is included as

$$
\mathrm{Q\ bytes} = 2 n_q d.
$$

The modeled arithmetic intensity is the ratio

$$
I(g) = \frac{\mathrm{FLOPs}}{\mathrm{KV\ bytes} + \mathrm{Q\ bytes}}.
$$

Increasing $g$ lowers the key/value bandwidth cost and increases the modeled intensity.

## Task

Implement `decode_arithmetic_intensity(g, n_q, d, seq_len)`.

The function takes:
- `g`: the GQA group factor.
- `n_q`: number of query heads.
- `d`: head dimension.
- `seq_len`: cached sequence length.

Return the floating point FLOP/byte arithmetic intensity from the model above. The function should support `g` values that divide `n_q`.

## Example

```python
x = decode_arithmetic_intensity(4, 32, 128, 2048)
# x is the modeled FLOP/byte value for a 4-way grouped query layout
```

## What the gate checks

The gate computes the analytic model independently and compares the returned value for several group factors, including $g \in \{1,4,8,n_q\}$. The reported metric `modeled_arith_intensity` is the fraction of cases whose absolute error is at most $10^{-6}$.
