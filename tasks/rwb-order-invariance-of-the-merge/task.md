## Context

Distributed numerical systems often compute partial results on different workers and then merge those chunks into one result. A merge implementation should not depend on the order in which workers finish.

For a collection of partial vectors $p_1, p_2, \dots, p_S \in \mathbb{R}^d$, the mathematical merge is

$$
m = \sum_{s=1}^{S} p_s .
$$

Floating point addition is not associative because rounding is applied after each operation. Therefore, evaluating

$$
((p_1 + p_2) + p_3) + \dots + p_S
$$

can produce a different result from another ordering. Production numerical libraries avoid this source of nondeterminism by using a stable reduction strategy with a fixed accumulation order and higher precision accumulation.

## Task

Implement `merge_partials(partials)`.

The input `partials` is a list of NumPy arrays. Each array contains one worker's partial result and all arrays have the same shape. Return a pair:

```python
(given_order, reversed_order)
```

where both outputs are NumPy arrays containing the merged result.

The merge must be deterministic regardless of the order of `partials`. Use a float64 accumulation strategy and define a canonical merge order internally. The returned arrays must have dtype `float64`.

The first returned value must represent merging the input chunks. The second returned value must represent merging the same chunks when the input order is reversed. Both values should be identical within floating point precision.

## Example

```python
import numpy as np

partials = [
    np.array([1e16, 2.0]),
    np.array([-1e16, 3.0]),
    np.array([4.0, 5.0]),
]

a, b = merge_partials(partials)

# a and b contain the same deterministic merge result
```

## What the gate checks

The gate builds several sets of partial vectors and computes the reference result by merging all chunks with a NumPy float64 oracle.

The `max_abs_err` metric measures the largest absolute difference between the learner result and the oracle result. It must satisfy

$$
\max_i |x_i - y_i| < 10^{-10}.
$$

The `order_max_abs_err` metric measures the largest absolute difference between the learner's given-order and reversed-order outputs. It must satisfy

$$
\max_i |x_i - y_i| < 10^{-12}.
$$

A direct floating point accumulation in input order can fail because different worker completion orders produce different rounding paths.
