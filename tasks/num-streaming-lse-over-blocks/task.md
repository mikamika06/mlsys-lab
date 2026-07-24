## Context

The log-sum-exp of a vector $x \in \mathbb{R}^n$ is

$$
\operatorname{LSE}(x) = \log \sum_{i=1}^{n} e^{x_i} = m + \log \sum_{i=1}^{n} e^{x_i - m}, \qquad m = \max_i x_i ,
$$

where subtracting the max before exponentiating keeps every term in
$(0, 1]$ and avoids overflow. This batched form needs the whole vector in
memory to find $m$ before it can process a single term.

A **streaming** version processes $x$ one block at a time, never holding more
than one block plus a running $(m, s)$ pair — the same trick used by
flash-attention–style online softmax. When a new block arrives, its max might
exceed the running max, so the running accumulator must be *rescaled* before
adding the new block's contribution:

$$
m' = \max(m, m_{\text{block}}), \qquad
s' = s \cdot e^{\,m - m'} + \sum_{x_i \in \text{block}} e^{\,x_i - m'} .
$$

Starting from $m = -\infty,\, s = 0$ and folding in every block in sequence,
the final $\operatorname{LSE}(x)$ is $m + \log s$ after the last block —
identical to the batched result, but computed with only one block resident
at a time.

## Task

Implement `streaming_lse(x, block_size)`:

```python
def streaming_lse(x: np.ndarray, block_size: int) -> float:
    ...
```

`x` is a 1-D `float64` array of length $n$; `block_size` is a positive
integer that does not necessarily divide $n$ (the last block may be
shorter). Process `x` **sequentially in blocks of size `block_size`**,
maintaining a running max $m$ and running accumulator $s$ with the rescaling
update above, and return the final $m + \log s$ as a Python `float`. Do not
compute the log-sum-exp over the whole array in one call — the running
$(m, s)$ update across blocks is the point of the exercise.

## Example

```python
import numpy as np
x = np.array([1.0, 2.0, 3.0, 900.0, -5.0, 0.0])
streaming_lse(x, block_size=2)
# ~= 900.0  (dominated by the single large entry, computed without overflow)
```

## What the gate checks

Two gates. The grader compares your result against a batched stable
log-sum-exp reference (`m = x.max()`, then `m + log(sum(exp(x - m)))`) with

$$
\mathrm{rel\_err} = \frac{|\widehat{\operatorname{LSE}} - \operatorname{LSE}_{\text{ref}}|}{|\operatorname{LSE}_{\text{ref}}| + 10^{-12}} ,
$$

requiring $\mathrm{rel\_err} \le 10^{-12}$. A Python-level line tracer also
counts source lines executed inside your function across all test cases
(`op_count`); the gate requires $\mathrm{op\_count} \ge 1000$. Test arrays
are thousands of elements long with block sizes that leave a remainder, so a
genuine per-block Python loop emits thousands of line events, while
collapsing everything into one vectorized call over the whole array (ignoring
`block_size`) emits only a handful — it would pass the accuracy gate but fail
`op_count`.
