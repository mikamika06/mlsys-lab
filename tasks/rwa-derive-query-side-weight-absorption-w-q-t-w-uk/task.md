## Context

Attention-style scoring often contains a query projection followed by an up-projection
that maps a context representation into the same scoring space. For query vector
$x \in \mathbb{R}^{d_{\text{in}}}$ and context vector $c \in \mathbb{R}^{d_{\text{out}}}$,
a score can be written as

$$
(W_Q x)^\top (W_{UK} c).
$$

Using matrix algebra, the up-projection can be absorbed into the query-side
weight:

$$
(W_Q x)^\top (W_{UK} c)
=
x^\top W_Q^\top W_{UK} c.
$$

The matrix

$$
W_{\text{absorbed}} = W_Q^\top W_{UK}
$$

allows a production implementation to avoid materializing the projected context
vector $W_{UK}c$ for every query. The score can instead be computed as

$$
(W_{\text{absorbed}}^\top x)^\top c
$$

or equivalently by applying the absorbed transformation to the query.

## Task

Implement `absorb_query_weight(W_Q, W_UK)`:

```python
def absorb_query_weight(W_Q: np.ndarray, W_UK: np.ndarray) -> np.ndarray:
    ...
```

The function receives two floating point matrices:

- `W_Q` with shape $(d_{\text{head}}, d_{\text{in}})$
- `W_UK` with shape $(d_{\text{head}}, d_{\text{out}})$

Return the absorbed query-side matrix

$$
W_Q^\top W_{UK}
$$

with shape $(d_{\text{in}}, d_{\text{out}})$ and dtype `float64`.

Use NumPy matrix operations rather than explicit Python loops.

## Example

```python
import numpy as np

W_Q = np.array([[1.0, 2.0], [3.0, 4.0]])
W_UK = np.array([[5.0], [6.0]])

W_absorbed = absorb_query_weight(W_Q, W_UK)

# [[23.]
#  [34.]]
```

## What the gate checks

The gate computes a NumPy reference using the original scoring path in float64:

$$
(W_Q x)^\top(W_{UK} c)
$$

for several generated query and context vectors. It then evaluates the student's
absorbed matrix by computing the equivalent absorbed path without constructing
the full projected key representation.

The reported metric is `max_abs_err`, the largest absolute difference between
the oracle scores and the absorbed scores. The value must be less than
$10^{-4}$.
