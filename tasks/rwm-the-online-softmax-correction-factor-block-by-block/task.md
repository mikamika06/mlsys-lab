## Context

Production attention kernels cannot always materialize a full softmax vector. They
process score blocks and maintain an equivalent running softmax state.

For a sequence of scores $x_1,\dots,x_n$, the stable softmax denominator uses a
running maximum $m$ and a scaled sum $l$:

$$
l = \sum_i e^{x_i-m}.
$$

When a new block changes the maximum from $m_{\text{old}}$ to
$m_{\text{new}}$, all previous contributions must be rescaled by the correction
factor

$$
e^{m_{\text{old}}-m_{\text{new}}}.
$$

For values $v_i$, the weighted accumulator is updated in the same way:

$$
a_{\text{new}} =
a_{\text{old}} e^{m_{\text{old}}-m_{\text{new}}}
+
a_{\text{block}} e^{m_{\text{block}}-m_{\text{new}}}.
$$

This keeps a block-by-block computation mathematically equivalent to computing
one complete softmax.

## Task

Implement `online_softmax_blocks(scores, values, block_size)`:

```python
def online_softmax_blocks(
    scores: np.ndarray,
    values: np.ndarray,
    block_size: int
) -> np.ndarray:
    ...
```

`score` inputs are a one-dimensional array of logits and `values` is a
one-dimensional array of the same length. Split the sequence into consecutive
blocks of size `block_size`.

Return a `float64` NumPy array with one row per block. Each row must contain:

$$
[m,\ l,\ a]
$$

where $m$ is the running maximum after the block, $l$ is the running scaled
softmax denominator, and $a$ is the running weighted accumulator.

The first block should initialize the running state correctly. Use only the
online update rule; do not compute the full softmax over all elements at once.

## Example

```python
import numpy as np

scores = np.array([1.0, 2.0, 10.0, 3.0])
values = np.array([2.0, 1.0, 4.0, 5.0])

out = online_softmax_blocks(scores, values, 2)

# rows are:
# [running max, running denominator, running accumulator]
```

## What the gate checks

The gate computes the block updates with an independent NumPy oracle and compares
every emitted $(m,l,a)$ state using maximum absolute error.

The metric must satisfy

$$
\max |y_{\text{submission}}-y_{\text{oracle}}| \le 10^{-10}.
$$

A solution that omits the correction factor
$e^{m_{\text{old}}-m_{\text{new}}}$ will fail when a later block contains a
larger maximum.
