## Context

Online softmax computation stores a partial attention row using three values:

$$
m = \max_i x_i,
$$

$$
l = \sum_i e^{x_i-m},
$$

and

$$
o = \sum_i e^{x_i-m} v_i,
$$

where $x_i$ are logits and $v_i$ are value vectors. The normalized output is

$$
\mathrm{softmax}(x)V = \frac{o}{l}.
$$

When a sequence is processed in chunks, each chunk produces its own $(m,l,o)$ state. The states can be merged without storing all logits.

For two partial states, let

$$
m = \max(m_1,m_2).
$$

The combined normalization factor is

$$
l = l_1 e^{m_1-m} + l_2 e^{m_2-m},
$$

and the combined output accumulator is

$$
o = o_1 e^{m_1-m} + o_2 e^{m_2-m}.
$$

The final attention output is obtained from $o/l$.

## Task

Implement `merge_mlo(state1, state2)`:

```python
def merge_mlo(state1: tuple[float, float, list[float]], state2: tuple[float, float, list[float]]) -> tuple[float, float, list[float]]:
    ...
```

Each input state is a tuple `(m, l, o)`:

- `m` is a scalar maximum logit.
- `l` is a scalar normalization accumulator.
- `o` is a list of floats containing the weighted value accumulator.

Return the merged state `(m, l, o)` using the closed-form merge equations. Do not reconstruct the original logits.

The returned `o` must be a list with the same shape as the input accumulators.

## Example

```python

a = (2.0, 1.5, [1.0, 3.0])
b = (4.0, 2.0, [5.0, 1.0])

m, l, o = merge_mlo(a, b)
out = o / l
```

The value `out` represents the normalized result of combining the two partial softmax computations.

## What the gate checks

The gate builds partial states from halves of real logits and values using Python softmax calculations. It computes the oracle result by evaluating the full-row softmax times values directly, then compares it with the merged state output.

The metric is

$$
\max_i |y_i-\hat{y}_i|,
$$

where $y$ is the Python oracle output and $\hat{y}$ is the output from the merged $(m,l,o)$ state. The maximum absolute error must be at most $10^{-6}$.
