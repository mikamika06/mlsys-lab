## Context

Attention mechanisms assign a score to each key position when producing a query
representation. ALiBi adds a distance-dependent linear bias to attention logits.
For a query at position $q$ and key at position $k$, the bias for a head with
slope $m$ is

$$
b(q,k) = -m(q-k),
$$

where larger distances receive smaller logits.

The resulting attention weights are obtained by the softmax operation

$$
p(q,k) =
\frac{\exp(b(q,k))}
{\sum_{j=0}^{q}\exp(b(q,j))}.
$$

A locality metric can measure how far attention looks by computing the expected
normalized distance

$$
L =
\frac{1}{N}
\sum_q
\frac{\sum_k p(q,k)(q-k)}
{q+1}.
$$

Smaller values indicate stronger local attention. ALiBi keeps this distance
relationship linear as sequence length grows, which allows it to extrapolate
beyond the training context length without changing its bias rule.

## Task

Implement `alibi_extrapolation_metric`:

```python
def alibi_extrapolation_metric(
    num_heads: int,
    trained_len: int,
    extra_len: int,
) -> float:
    ...
```

Return the average ALiBi locality metric for query positions beyond the trained
length. The evaluated query positions are

$$
q = \text{trained\_len}, \dots, \text{trained\_len}+\text{extra\_len}-1.
$$

Use one ALiBi head slope per head:

$$
m_h = 2^{-\frac{h+1}{\text{num\_heads}}}.
$$

For every head and query position, compute softmax attention over all previous
positions including the query itself, then average the normalized expected
distance. Return a Python `float`.

## Example

```python
score = alibi_extrapolation_metric(2, 8, 4)
# score is a float locality score for positions 8 through 11
```

## What the gate checks

The gate recomputes the metric with a NumPy reference implementation of the
ALiBi equations. The returned value is compared using relative error
$\mathrm{rel\_err}$ and must satisfy

$$
\mathrm{rel\_err} \le 10^{-2}.
$$

A solution that uses a different bias rule or ignores extrapolated positions
will produce a different locality score.
