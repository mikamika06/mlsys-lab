## Context

Grouped-query attention (GQA) reduces memory usage by sharing key and value heads
between multiple query heads. If there are $h_q$ query heads and $h_{kv}$ key/value
heads, each key/value head is assigned to

$$r = \frac{h_q}{h_{kv}}$$

query heads. The expanded key/value tensor must repeat each original head
contiguously.

For a key/value tensor $K \in \mathbb{R}^{b \times h_{kv} \times s \times d}$,
the expanded tensor has shape

$$K' \in \mathbb{R}^{b \times h_q \times s \times d}.$$

The head mapping is

$$K'_{:,\,j,\,:\,,\, :} = K_{:,\,\lfloor j/r \rfloor,\,:\,,\, :}.$$

This means head order changes from

$$[0, 1, \dots, h_{kv}-1]$$

to

$$[\underbrace{0,\dots,0}_{r\text{ times}},
\underbrace{1,\dots,1}_{r\text{ times}}, \dots].$$

A common bug is using a tiling operation that creates

$$[0,1,\dots,h_{kv}-1,0,1,\dots,h_{kv}-1]$$

instead of repeating each head group.

## Task

Implement `expand_kv_heads(kv, num_q_heads)`:

```python
def expand_kv_heads(kv: np.ndarray, num_q_heads: int) -> np.ndarray:
    ...
```

The input `kv` is a NumPy array with shape $(b, h_{kv}, s, d)$. The value
`num_q_heads` is a positive multiple of the number of key/value heads.

Return a new array with shape $(b, num_q_heads, s, d)$ where each key/value head
is repeated the required number of times for grouped-query attention. Use NumPy
operations and preserve the input dtype.

## Example

```python
import numpy as np

kv = np.array(
    [[[[1.0]], [[2.0]]]]
)

out = expand_kv_heads(kv, 4)

# Head values are [1, 1, 2, 2].
# The shape is (1, 4, 1, 1).
```

## What the gate checks

The gate builds several GQA tensors and computes the expected expansion using a
NumPy reference implementation with `np.repeat`. The candidate output is compared
using

$$\max_i |x_i - y_i|.$$

The value must be less than $10^{-5}$. Implementations that use the wrong
interleaving order fail because the expanded head sequence differs.
