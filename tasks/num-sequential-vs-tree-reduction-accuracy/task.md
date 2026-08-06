## Context

Floating point addition is not associative. For real numbers,

$$
(a+b)+c = a+(b+c),
$$

but floating point arithmetic rounds intermediate results, so the equality may not hold after every operation. A long sequential reduction can accumulate rounding error because each new value is added to an already rounded partial sum.

A tree reduction changes the order of operations. It repeatedly combines nearby pairs:

$$
s_0 = a_0 + a_1,\quad s_1 = a_2 + a_3,\quad \dots
$$

and continues until one value remains. This creates a shallower addition tree and can reduce accumulated error compared with a left-to-right reduction.

The accuracy of a result $x$ can be measured against a higher precision reference value $r$ using relative error:

$$
\mathrm{rel\_err}(x,r)=\frac{|x-r|}{|r|+\epsilon}.
$$

The reference value in this task is computed using Python float64 accumulation.

## Task

Implement `tree_sum(values)`:

```python
def tree_sum(values: list[float]) -> float:
    ...
```

The function receives a list of floats of `float32` values and returns a `float32` sum using a pairwise tree reduction. Do not convert the full input to `float64`. The intermediate additions should remain in `float32`.

The reduction should continue pairing values until a single value remains. If a reduction level has an odd number of values, you may handle the remaining value in a way that preserves the tree structure.

## Example

```python

values = [1e8, 1, 1, 1, -1e8]

result = tree_sum(values)
```

A sequential float32 reduction may lose small contributions depending on the order of additions. A tree reduction changes the order and can produce a result closer to the float64 reference.

## What the gate checks

The gate computes a float64 Python reduction as the oracle. It also computes the error of a sequential float32 reduction and the error of the submitted tree reduction on the same fixture.

The reported `rel_err` is the ratio:

$$
\frac{\mathrm{rel\_err}_{tree}}{\mathrm{rel\_err}_{sequential}+\epsilon}.
$$

The submission passes when the tree reduction error is at most half of the sequential reduction error. This requires improving the numerical behavior rather than only returning the same sequential algorithm.
