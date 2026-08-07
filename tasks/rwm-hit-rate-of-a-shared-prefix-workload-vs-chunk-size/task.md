## Context

Production inference systems often reuse cached key-value states from shared request prefixes. A request with a shared prefix of length $p$ tokens can only reuse complete chunks of size $c$.

The number of reusable tokens for one request is

$$
r(p, c) = \left\lfloor \frac{p}{c} \right\rfloor c .
$$

Given request prefix lengths $p_1, p_2, \dots, p_n$, the average hit rate is the fraction of tokens that can be reused:

$$
H = \frac{1}{n}\sum_{i=1}^{n}\frac{r(p_i,c)}{p_i}.
$$

This metric models how chunk size affects cache efficiency. Smaller chunks usually increase reuse because fewer prefix tokens are wasted at chunk boundaries.

## Task

Implement `prefix_chunk_hit_rate(prefix_lengths, chunk_size)`:

```python
def prefix_chunk_hit_rate(prefix_lengths: list[int], chunk_size: int) -> float:
    ...
```

The function receives a list of floats of positive integer prefix lengths and a positive integer chunk size. Return the mean reusable fraction as a Python `float`.

The implementation should compute the analytic reuse fraction directly. Do not simulate token insertion or build per-token representations.

## Example

```python

prefixes = [10, 11, 15]
rate = prefix_chunk_hit_rate(prefixes, 4)

# reusable tokens:
# 8, 8, 12
# fractions:
# 0.8, 8/11, 0.8
```

## What the gate checks

The gate computes an oracle result using Python from the mathematical definition

$$
\frac{1}{n}\sum_i
\frac{\left\lfloor p_i / c \right\rfloor c}{p_i}.
$$

The submitted implementation must match the oracle with relative error $\mathrm{rel\_err} \le 10^{-6}$ over several prefix distributions and chunk sizes.
