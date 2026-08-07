## Context

Grouped-query attention (GQA) reduces memory usage by sharing key and value heads
between multiple query heads. If there are $H_q$ query heads and $H_{kv}$ key/value
heads, each KV head is assigned to a group of query heads.

The mapping is an interleave operation. The repetition factor is

$$r = \frac{H_q}{H_{kv}}.$$

For a tensor of KV heads with shape $(B, H_{kv}, T, d)$, the expanded tensor used
by attention should have shape $(B, H_q, T, d)$:

$$
K_{\mathrm{expanded}}[:, i, :, :] =
K_{\mathrm{kv}}[:, \lfloor i / r \rfloor, :, :].
$$

This means each KV head is repeated consecutively. For example, with
$H_q=8$ and $H_{kv}=2$, the head mapping is

$$[0, 0, 0, 0, 1, 1, 1, 1].$$

Using tiling creates a different mapping, such as
$[0,1,0,1,0,1,0,1]$, which silently mixes groups.

## Task

Implement `expand_kv_heads(kv, num_query_heads)`:

```python
def expand_kv_heads(kv: list[list[list[list[float]]]], num_query_heads: int) -> list[list[list[list[float]]]]:
    ...
```

The input `kv` is a list with shape $(B, H_{kv}, T, d)$.
Return a new array with shape $(B, H_q, T, d)$ by repeating every KV head the
required number of times. Assume that `num_query_heads` is divisible by
`kv.shape[1]`.

The output must preserve the input values and dtype.

## Example

```python

kv = [[[[1]], [[2]]]]
out = expand_kv_heads(kv, 6)

# Head values are [1, 1, 1, 2, 2, 2]
# out has shape (1, 6, 1, 1)
```

## What the gate checks

The gate compares the submitted implementation against a Python oracle that uses
the required head mapping with list repetition. The relative error

$$
\mathrm{rel\_err} =
\frac{\lVert y_{\mathrm{candidate}} - y_{\mathrm{reference}}\rVert_2}
{\lVert y_{\mathrm{reference}}\rVert_2 + 10^{-12}}
$$

must be less than $10^{-5}$. Implementations using list tiling fail because they
produce a different head-to-group assignment.
