## Context

Online softmax implementations often process a stream in blocks. A block is summarized by its maximum logit and a scaled sum of exponentials. A useful representation of a block is the pair

$$
(m, s)
$$

where $m$ is the maximum value in the block and

$$
s = \sum_i \exp(x_i - m).
$$

Two adjacent summaries can be merged without revisiting the original values. For summaries $(m_1, s_1)$ and $(m_2, s_2)$, the merge operator is

$$
m = \max(m_1, m_2),
$$

$$
s = s_1 \exp(m_1 - m) + s_2 \exp(m_2 - m).
$$

If the operator is implemented correctly, different parenthesizations of a sequence of blocks should produce numerically equivalent results. For example,

$$
((A \oplus B) \oplus C) \approx (A \oplus (B \oplus C)).
$$

This property is important for parallel reductions because workers may combine blocks in different tree shapes.

## Task

Implement `check_block_merge_associativity(rows)`.

The input is a 2-D NumPy array. Each row contains logits that are split into several consecutive blocks internally by the function.

The function must return a 1-D boolean NumPy array. For every row, return `True` if all tested merge bracketings of the row's blocks agree within absolute tolerance $10^{-6}$ for both components of the final summary $(m, s)$.

The function must:

- create between 2 and 5 consecutive blocks for each row,
- compute each block summary using the stable online-softmax representation,
- merge summaries using the associative merge operator,
- compare multiple reduction bracketings,
- return only the boolean result array.

Do not use the original logits after a block summary has been created.

## Example

```python
import numpy as np

x = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [1000.0, 999.0, 998.0, 997.0],
])

result = check_block_merge_associativity(x)

# result is:
# array([True, True])
```

## What the gate checks

The gate uses a NumPy oracle implementation of the same mathematical merge operator. It generates several deterministic random inputs and partitions each row into different numbers of blocks.

The returned boolean vector must exactly match the oracle result. The oracle evaluates multiple bracketings of each partition and marks a row as valid when all final summaries agree within $10^{-6}$.
