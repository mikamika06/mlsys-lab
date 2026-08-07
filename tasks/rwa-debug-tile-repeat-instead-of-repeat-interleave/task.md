## Context

Grouped-query attention (GQA) stores key and value tensors with fewer heads than
the query tensor. Each key/value head is shared by several query heads. If the
number of query heads is $H_q$ and the number of key/value heads is $H_{kv}$,
then each key/value head is repeated

$$r = \frac{H_q}{H_{kv}}$$

times.

The expansion must preserve grouping. For a key/value tensor with head axis
values $[K_0, K_1, \dots, K_{H_{kv}-1}]$, the expanded order is

$$[K_0, \underbrace{K_0,\dots,K_0}_{r\ \text{copies}},
K_1,\underbrace{K_1,\dots,K_1}_{r\ \text{copies}},\dots].$$

A common implementation mistake is using tiling. `tile` repeats the whole head
sequence, producing

$$[K_0,K_1,\dots,K_0,K_1,\dots]$$

which assigns query heads to the wrong key/value groups.

## Task

Implement `expand_kv_heads(kv, n_query_heads)`:

```python
def expand_kv_heads(kv: list[list[list[list[float]]]], n_query_heads: int) -> list[list[list[list[float]]]]:
    ...
```

The input `kv` is a 4-D list with shape
$(B, H_{kv}, S, D)$ representing batch size, key/value heads, sequence length,
and head dimension. Return a new array with shape
$(B, H_q, S, D)$ where $H_q$ equals `n_query_heads`.

Each key/value head must be repeated contiguously along axis $1`. The returned
array should use `float64` values.

## Example

```python

kv = [[[[1.0], [2.0]],
      [[3.0], [4.0]]]]

out = expand_kv_heads(kv, 4)
# out[:, :, :, 0] contains:
# [
#   [[1, 2], [1, 2], [3, 4], [3, 4]]
# ]
```

## What the gate checks

The gate computes a Python oracle using list repetition along the head axis and
compares the submitted implementation against it with maximum absolute error
$\le 10^{-5}$.

An implementation using list extension produces a different head ordering and fails
because query heads attend to the wrong key/value groups.
