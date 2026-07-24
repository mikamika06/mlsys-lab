## Context

Reduction operations combine many values into one result. For an associative operator such as addition, the order of grouping changes the reduction tree but not the mathematical result.

For values $x_0, x_1, \dots, x_{N-1}$, a pairwise tree reduction combines adjacent pairs at each level:

$$
y_i = x_{2i} + x_{2i+1}.
$$

The next level reduces the intermediate values again until a single value remains. A balanced binary tree has depth

$$
\lceil \log_2 N \rceil .
$$

This structure is important for parallel systems because independent pairs can be processed simultaneously.

## Task

Implement `tree_reduce(values)`.

The function receives a Python list of numeric values and returns a tuple:

```python
(result, trace)
```

`result` must be the sum of all values. `trace` must describe the reduction levels. It must be a list where each element represents one tree level, and each level contains the number of values present after that reduction step.

Use pairwise reduction. At each level, combine adjacent elements. If a level has an odd number of elements, carry the final unpaired element to the next level unchanged.

The returned trace must contain exactly one entry per reduction level. For a single input value, the trace is an empty list.

## Example

```python
result, trace = tree_reduce([1.0, 2.0, 3.0, 4.0, 5.0])

# result == 15.0
# trace == [[3], [2], [1]]
```

The trace records the number of values remaining after each pairwise reduction level.

## What the gate checks

The gate computes the expected reduction using an independent tree reduction algorithm. The returned result must have maximum absolute error below $10^{-9}$ compared with the oracle result.

The gate also checks that the trace describes a balanced tree. Its depth, measured as the number of trace levels, must equal

$$
\lceil \log_2 N \rceil .
$$

A sequential left-to-right reduction has the correct value but does not satisfy the required tree structure.
