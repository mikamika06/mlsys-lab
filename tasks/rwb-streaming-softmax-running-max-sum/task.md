## Context

The numerically stable softmax of $x\in\mathbb{R}^n$ is

$$
\mathrm{softmax}(x_i) = \frac{e^{x_i-m}}{\sum_j e^{x_j-m}}, \qquad m=\max_j x_j,
$$

which requires knowing the whole vector's max before exponentiating
anything. When $x$ arrives in pieces (too long to hold at once, or
produced incrementally), the **streaming softmax** algorithm processes
chunks one at a time, maintaining a running max $m$ and running sum $l$
that always represent the exact statistics of every element seen so far:

$$
m_{\text{new}} = \max(m,\ \max(\text{chunk})), \qquad
l \leftarrow l\cdot e^{\,m-m_{\text{new}}} + \sum_{x_i \in \text{chunk}} e^{\,x_i - m_{\text{new}}}, \qquad
m \leftarrow m_{\text{new}}.
$$

The rescale-by-$e^{m-m_{\text{new}}}$ step is what keeps $l$ correct even
after the running max changes partway through: without it, terms
exponentiated against a since-superseded (too-small) max would be
systematically too large. After the final chunk, every element's output
is $e^{x_i - m}/l$ — identical to the direct formula above, just computed
without ever needing the whole vector's true max in advance.

## Task

Implement `streaming_softmax(scores, chunk_size)`:

```python
def streaming_softmax(scores: np.ndarray, chunk_size: int) -> np.ndarray:
    ...
```

- `scores`: 1-D array of raw scores (may be `float32`, may contain very
  large or very negative values).
- `chunk_size`: process at most this many elements per step (the final
  chunk may be smaller if `chunk_size` doesn't evenly divide
  `len(scores)`).

Sweep `scores` in chunks using the running-max/running-sum recurrence
above, then return the full softmax vector (`float64`, same shape as
`scores`) using the final `(m, l)`.

## Example

```python
import numpy as np

x = np.array([1000.0, 1.0, 2.0, -1000.0], dtype=np.float32)
streaming_softmax(x, chunk_size=2)
# array([1., 0., 0., 0.])  -- matches a direct stable softmax over the
# whole vector; a naive (non-online) softmax computed chunk-by-chunk
# without ever revisiting the running max would get this wrong.
```

## What the gate checks

The gate loads a fixed 37-element `float32` fixture (`scores.npy`) with
wide dynamic range and two deliberate outliers (`1000.0` and `-1000.0`),
run at several `chunk_size` values including one that doesn't evenly
divide the length (ragged last chunk) and one equal to the whole vector,
plus several seeded synthetic vectors of varying length, chunk size, and
magnitude. For each, the oracle computes a direct numerically stable
softmax in `float64`, independent of any chunking.

Your output is compared element-for-element (`max_abs_err`, threshold
`1e-6`). Forgetting the rescale step when the running max updates,
resetting `m`/`l` between chunks instead of carrying them forward, or
mishandling the ragged final chunk will diverge sharply on the
wide-dynamic-range fixture.
