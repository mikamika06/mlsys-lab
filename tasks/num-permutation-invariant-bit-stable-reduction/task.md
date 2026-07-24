## Context

Floating-point addition is not associative. For values $a$, $b$, and $c$ it is
possible that

$$
(a+b)+c \ne a+(b+c)
$$

because each intermediate operation rounds to the nearest representable
floating-point value.

A reduction over an array therefore depends on the order in which elements are
combined. If the same values are provided in different permutations, a normal
left-to-right sum can produce different bits.

This task uses a canonical reduction strategy. The input values are first ordered
by IEEE-754 total ordering of their `float64` bit patterns. The ordered sequence
is then reduced with a fixed pairwise tree:

$$
R_{k+1,i} = R_{k,2i} + R_{k,2i+1}
$$

until one value remains. If the input contains NaNs, the function must return the
first NaN in the canonical order instead of performing arithmetic on the NaN
values. This makes NaN propagation deterministic.

## Task

Implement `stable_sum(values)`:

```python
def stable_sum(values: np.ndarray) -> np.float64:
    ...
```

The input is a one-dimensional NumPy array. Convert values to `float64`, apply
the canonical bit ordering described above, and return the deterministic
reduction result as `np.float64`.

The function must produce the same bit pattern for every permutation of the same
multiset of inputs. Signed zero values, infinities, and NaNs are part of the
contract.

## Example

```python
import numpy as np

x = np.array([1e20, 1.0, -1e20, 0.0], dtype=np.float64)

a = stable_sum(x)
b = stable_sum(x[[2, 0, 3, 1]])

# a and b have identical float64 bit patterns.
```

## What the gate checks

The gate creates several random permutations of arrays containing ordinary
numbers, cancellation cases, infinities, signed zeros, and NaNs.

It computes the expected behavior using the same canonical-order and fixed-tree
algorithm inside the grader. The returned `float64` bit pattern must match the
oracle result, and the result must remain identical across all tested
permutations.
