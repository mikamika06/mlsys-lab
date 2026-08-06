## Context

Online softmax algorithms maintain a running maximum and a running weighted output while processing blocks of attention scores.

Suppose a previous block has maximum score $m_{\mathrm{old}}$, normalization factor $l_{\mathrm{old}}$, and output accumulator $O_{\mathrm{old}}$. A new block has maximum score $m_{\mathrm{blk}}$, normalization factor $l_{\mathrm{blk}}$, and output accumulator $O_{\mathrm{blk}}$.

When the maximum changes, the old accumulator was computed at a different exponential scale. The new maximum is

$$
m_{\mathrm{new}} = \max(m_{\mathrm{old}}, m_{\mathrm{blk}}).
$$

The old contribution must be rescaled by

$$
\alpha = \exp(m_{\mathrm{old}} - m_{\mathrm{new}}),
$$

and the block contribution by

$$
\beta = \exp(m_{\mathrm{blk}} - m_{\mathrm{new}}).
$$

The updated normalization is

$$
l_{\mathrm{new}} = \alpha l_{\mathrm{old}} + \beta l_{\mathrm{blk}},
$$

and the updated output is

$$
O_{\mathrm{new}} =
\frac{\alpha l_{\mathrm{old}} O_{\mathrm{old}} + \beta l_{\mathrm{blk}} O_{\mathrm{blk}}}
{l_{\mathrm{new}}}.
$$

A common FlashAttention implementation bug is forgetting the factor $\exp(m_{\mathrm{old}}-m_{\mathrm{new}})$ on the old output contribution. This produces incorrect outputs whenever the running maximum increases.

## Task

Implement `online_softmax_update`:

```python
def online_softmax_update(
    m_old,
    l_old,
    O_old,
    m_block,
    l_block,
    O_block,
):
    ...
```

The function must return `(m_new, l_new, O_new)` using the numerically stable update equations above.

`O_old` and `O_block` are list of floats with the same shape. Return `O_new` as a list with dtype `float64`.

## Example

```python

m_new, l_new, O_new = online_softmax_update(
    1.0,
    2.0,
    [3.0, 4.0],
    3.0,
    1.5,
    [5.0, 6.0],
)

# m_new == 3.0
# l_new and O_new use the rescaled online softmax update.
```

## What the gate checks

The gate computes a reference result from the mathematical online softmax update using Python. It compares the returned output values with the reference using maximum absolute error:

$$
\mathrm{max\_abs\_err} = \max_i |x_i - y_i|.
$$

The result must satisfy $\mathrm{max\_abs\_err} < 10^{-5}$. Inputs include cases where the maximum changes and the missing rescale term causes a measurable error.
